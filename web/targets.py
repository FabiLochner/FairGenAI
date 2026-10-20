"""Target distributions and their translation into bounded-repetition deadlines.

Algorithm 1 of Cheng et al. enforces <cgf2 | cgf1 <= cg> []<>_{<=beta}-fairness: every
group value k must appear at least once in every window of beta_k related images.

Key point: such a deadline only guarantees a MINIMUM SHARE for k, F(k) >= 1/beta_k
(e.g. beta_k = 4 -> at least 25%). It can never limit a group to "at most" a share.
A target distribution r* is therefore translated into minimum shares, one per group value.
"""

import math    # math.floor = round down; turns a share into an integer deadline
import random  # weighted random choice when several group values are equally urgent


def check_distribution(dist, values, name):
    """Validate that a distribution from the config covers exactly the group values and sums to 1."""
    # Group values that the config forgot to give a share to (e.g. "female" missing)
    missing = set(values) - set(dist)
    # Keys in the config that are not group values (e.g. a typo like "femal")
    unknown = set(dist) - set(values)
    # Both cases mean the config does not describe the grouping function -> stop early
    if missing or unknown:
        raise ValueError(f"{name}: missing values {sorted(missing)}, unknown values {sorted(unknown)}")
    # A distribution has to sum to 1 (small tolerance for rounded census shares)
    total = sum(dist.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name}: shares sum to {total}, not 1")
    # Return a copy so later changes do not modify the loaded config
    return dict(dist)


def deadlines_from_target(target):
    """Translate a target distribution into a beta vector (one deadline per group value).

    beta_k = 1/r_k rounded down guarantees F(k) >= 1/beta_k >= r_k.
    Example: r = {"male": 0.9, "female": 0.1}
      female: 1/0.1 = 10 -> beta = 10 (at least one woman in every 10 images)
      male:   1/0.9 = 1.1 -> rounded down 1 -> "every image male", impossible together
              with the female deadline -> men get no deadline (left to the generator).

    Each deadline reserves a share 1/beta_k of all images. Values are added in ascending
    order of their target share as long as the reserved shares together stay <= 100%;
    the remaining (largest) values get no deadline.
    TODO: with more than two values that have a deadline, "<= 100% reserved" is not always
    enough for all deadlines to be met; a stricter check (<= 5/6) is needed then.

    Returns (beta, unconstrained).
    """
    # beta: group value -> deadline; unconstrained: values that get no deadline at all
    beta, unconstrained = {}, []
    # Smallest target shares first: these are the groups that most need a guaranteed minimum share
    for value in sorted(target, key=lambda v: target[v]):
        # Target share r_k of this group value, e.g. 0.10 for "female"
        share = target[value]
        # Deadline beta_k = 1/r_k rounded down, e.g. 1/0.1003 = 9.97 -> 9 -> at least one in every 9 images.
        # Rounding down makes the window shorter, so the guaranteed minimum share
        # (1/9 = 11.1%) is never below the target share (10.03%).
        # A share of 0 means "no minimum share wanted", so no deadline is computed.
        b = math.floor(1 / share) if share > 0 else None
        # Share of all images already reserved by the deadlines accepted so far
        reserved = sum(1 / x for x in beta.values())
        # Skip this value if
        #  - it wants no minimum share (share 0),
        #  - beta_k would be 1, i.e. *every* related image would have to show k, or
        #  - its reserved share 1/beta_k would push the total above 100%, so the deadlines could not all be met
        if b is None or b < 2 or reserved + 1 / b > 1 + 1e-9:
            unconstrained.append(value)
        else:
            # Accept the deadline for this group value
            beta[value] = b
    return beta, unconstrained


def build_plan(cfg, values):
    """Build the enforcement plan from the "fairness_enforcement" config block.

    Modes:
      cheng     -- uniform scalar beta for all values, as in the source paper
      parity    -- target r_k = 1/CG
      empirical -- target = reference (descriptive anchor) distribution
      maximin   -- target = maximin distribution r*
    """
    # The "target" block is optional; without it the server behaves like the original code
    target_cfg = cfg.get('target')
    if target_cfg is None or target_cfg.get('mode', 'cheng') == 'cheng':
        # Original behaviour: one scalar beta shared by all group values
        b = cfg['beta']
        # Lemma 2 of Cheng et al.: beta smaller than the number of values cannot be satisfied
        if b < len(values):
            raise ValueError("beta must be at least the number of group values (Lemma 2)")
        # Every group value gets the same deadline; no explicit target distribution exists
        return {'mode': 'cheng', 'target': None, 'beta': {v: b for v in values}, 'unconstrained': []}

    # One of the target-based modes
    mode = target_cfg['mode']
    if mode == 'parity':
        # Equal share for every group value, e.g. 0.5 / 0.5 for gender
        target = {v: 1 / len(values) for v in values}
    elif mode == 'empirical':
        # Descriptive anchor: the census / labour-statistics distribution for this condition
        target = check_distribution(target_cfg['reference']['distribution'], values, 'reference')
    elif mode == 'maximin':
        # Normative target r* derived via the difference principle (given explicitly in the config)
        target = check_distribution(target_cfg['maximin_distribution'], values, 'maximin_distribution')
    else:
        # Typo or unsupported mode in the config
        raise ValueError(f"Unknown target mode: {mode}")

    # Turn the target shares into deadlines (= minimum shares)
    beta, unconstrained = deadlines_from_target(target)
    return {'mode': mode, 'target': target, 'beta': beta, 'unconstrained': unconstrained}


class DeadlineScheduler:
    """Counter-based deadline scheduler (Algorithm 1, lines 1, 7-11, 13-16).

    c[k] is the number of related images within which k still has to appear, counting
    the next one. Only constrained values (those with a beta) carry a counter.
    """

    def __init__(self, beta, weights=None, rng=None):
        # Deadline per constrained group value, e.g. {"female": 10}
        self.beta = dict(beta)
        # Line 1 of Algorithm 1: every counter starts at its deadline
        self.counters = dict(beta)
        # Weights for breaking ties in line 9 (the target shares); uniform if none are given
        self.weights = weights or {v: 1.0 for v in beta}
        # Random source; can be replaced by a seeded random.Random for reproducible runs
        self.rng = rng or random
        # Number of missed deadlines (enforcement that the generator did not follow)
        self.violations = 0

    def select(self):
        """Return the value to enforce for the next related image, or None.

        Generalises line 8 to heterogeneous beta: enforcement is mandatory as soon as
        at least k values have a counter <= k. Among those, the most urgent (smallest
        counter) is served; ties are broken by drawing with the target weights instead
        of uniformly (line 9).
        """
        # Nothing to enforce if no group value has a deadline
        if not self.counters:
            return None
        # Lines 7-8: look from the largest possible counter down to 1, so a problem is detected as early as possible.
        # The outer max(..., 1) keeps k = 1 in the loop when every counter is 0 (missed deadlines),
        # otherwise range(0, 0, -1) would be empty and enforcement would silently stop.
        for k in range(max(max(self.counters.values()), 1), 0, -1):
            # Values whose deadline is at most k images away
            urgent = [v for v, c in self.counters.items() if c <= k]
            # If k or more values must appear within the next k images, there is no slack left:
            # the next image has to serve one of them
            if len(urgent) >= k:
                # Earliest deadline first: serve the value with the smallest counter
                most_urgent = min(self.counters[v] for v in urgent)
                # All values sharing that smallest counter are equally urgent
                candidates = [v for v in urgent if self.counters[v] == most_urgent]
                # Line 9, modified: draw among them weighted by the target share instead of uniformly
                return self.rng.choices(candidates, weights=[self.weights.get(v, 1.0) for v in candidates])[0]
        # No deadline is close enough yet: leave the user prompt untouched (minimal interference)
        return None

    def update(self, observed):
        """Update counters with the value realised in the image (lines 14-16).

        observed is None for images that do not count (no person, or the group value is
        not recognisable; "removal" in Definition 3 of the paper): they leave every
        counter unchanged.
        """
        # Images that do not count do not move any deadline
        if observed is None:
            return
        for v in self.counters:
            if v == observed:
                # Line 15: the value appeared, so its deadline starts over
                self.counters[v] = self.beta[v]
            else:
                # Line 16: one more image without this value
                self.counters[v] -= 1
                # Counter reached 0: the value did not appear within its deadline
                if self.counters[v] == 0:
                    self.violations += 1
                # Never go below 0: a missed deadline stays maximally urgent until the value appears
                self.counters[v] = max(self.counters[v], 0)
