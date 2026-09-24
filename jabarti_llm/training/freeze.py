"""
freeze.py -- finetune by choosing which of the model's own weights get to
move, rather than moving all of them or adding new ones
"""

def freeze_all(model):
    """Set every parameter's requires_grad to False.

    A blank slate: nothing in the model trains until something is
    explicitly unfrozen afterwards. Trainer._build_optimizer already skips
    any parameter with requires_grad=False (it has to, for LoRA's frozen
    base weights to work at all), so this alone is enough to stop the
    optimizer from touching anything here -- no other code needs to know
    freezing happened.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False
    

def unfreeze_tail(model, n_blocks: int = 2, unfreeze_lm_head: bool = True):
    """Re-enable gradients for the last `n_blocks` Transformer blocks, the
    final LayerNorm, and (by default) the output projection.
    """

    for block in model.blocks[-n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.ln_final.parameters():
        parameter.requires_grad = True

    if unfreeze_lm_head:
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

    