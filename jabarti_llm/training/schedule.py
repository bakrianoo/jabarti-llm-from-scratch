"""
schedule.py -- how the learning rate changes over a run
"""

import math

import torch


def cosine_with_warmup(optimizer, config):
    """Ramp up, then decay along a cosine curve.

    Two problems, one schedule:

    Warmup exists because the first steps are the most dangerous. Adam's
    running estimates start empty, so early updates are badly scaled, and a
    full learning rate applied to a randomly initialised model can push it
    somewhere it never recovers from. Starting near zero and ramping over
    warmup_steps avoids that.

    Decay exists because the step size that makes early progress is too coarse
    to settle into a minimum later. The cosine curve spends most of its time
    near the peak and eases off at the end, rather than dropping in steps.

    Returns a LambdaLR, so it multiplies whatever learning_rate the optimizer
    was built with.
    """

    def scale_for_step(step):
        
        # === warmup
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps

        # === cosine decay
        decay_steps = max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, (step - config.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # === floor
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, scale_for_step
    )