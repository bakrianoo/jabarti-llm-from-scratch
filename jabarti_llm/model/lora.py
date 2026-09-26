"""
lora.py -- Low-Rank Adaptation: finetune by adding a small detour, not by
moving the original weights
"""

import torch
import torch.nn as nn

# The four linear layers inside one attention block (attention.py). LoRA
# targets these because attention is where a model decides *what to attend
# to* -- the part most likely to need adjusting for a new task -- while
# leaving the feed-forward layers (where per-token computation happens)
# untouched. Nothing stops a script from passing a longer tuple in; this is
# just the common, well-tested default.

ATTENTION_PROJECTIONS = ("W_q", "W_k", "W_v", "W_o")


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int, alpha: int=16):
        super().__init__()
        self.base = base

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.r = r
        self.scale = alpha / r

        self.A = nn.Parameter(
            torch.empty(r, base.in_features)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, r)
        )

        # bound = 1 / sqrt(fan_in) = 1 / sqrt(512) ≈ 0.044
        # So every value in A is picked at random between -0.044 and +0.044.

        # What is a=5**0.5? --> It's a number, √5 ≈ 2.236 to adjust the formula
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        detour = (x @ self.A.T) @ self.B.T
        return self.base(x) + self.scale * detour

    def merged_weight(self):
        """The one weight matrix an ordinary nn.Linear would need to behave
        identically to this layer, right now.

        Example:

            B @ A has shape (512, 8) @ (8, 512) = (512, 512). That is the same shape as W.

            So you can add the detour straight into W:
            new W = old W + scale × (B @ A)

            
        """

        return self.base.weight + self.scale * (self.B @ self.A)

class TrainableRowsEmbedding(nn.Module):
    """A frozen embedding table where only a few chosen rows can learn.

    Pretraining never fed [SYS]/[USER]/[ASST] to the model, so their rows are
    still their random initial values. apply_lora freezes the whole table, so
    without this they would stay random for the entire finetune -- the very
    tokens the chat template is built from.
    """

    def __init__(self, base: nn.Embedding, token_ids):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False

        self.token_ids = list(token_ids)

        # Start from the current values, so the model's output is unchanged
        # until training moves them -- the same promise B=0 makes for LoRA.
        self.rows = nn.Parameter(
            base.weight.detach()[self.token_ids].clone()
        )

        # slot_of[id] = which row of self.rows holds that id, or -1 for "use
        # the base table". A buffer, so .to(device) carries it along.
        slot_of = torch.full((base.num_embeddings,), -1, dtype=torch.long)
        slot_of[self.token_ids] = torch.arange(len(self.token_ids))
        self.register_buffer("slot_of", slot_of, persistent=False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, input_ids):
        out = self.base(input_ids)                  # (B, T, d_model)
        slot = self.slot_of[input_ids]              # (B, T)
        is_trainable = (slot >= 0).unsqueeze(-1)    # (B, T, 1)

        # clamp so the -1 slots index something valid; torch.where throws
        # those rows away anyway.
        return torch.where(is_trainable, self.rows[slot.clamp(min=0)], out)


def train_token_rows(model, token_ids):
    """Let the embedding rows of `token_ids` learn, and nothing else in the
    table. Call after apply_lora (which freezes everything)."""

    model.embedding.token_embedding = TrainableRowsEmbedding(
        model.embedding.token_embedding, token_ids
    )
    return len(token_ids)

def merge_token_rows(model):
    """Write the trained rows back into an ordinary nn.Embedding.

    With tied weights, the embedding table IS lm_head's weight, so writing the
    new rows in would also change the output scores for those tokens --
    scores the model never trained against (lm_head stayed frozen). To keep
    the merged model behaving exactly like the trained one, lm_head gets its
    own copy of the original table first, and the config records the untie.
    """

    module = model.embedding.token_embedding
    if not isinstance(module, TrainableRowsEmbedding):
        return 0

    plain = module.base

    if model.lm_head.weight is plain.weight:
        model.lm_head.weight = nn.Parameter(plain.weight.detach().clone())
        model.config.tie_weights = False

    with torch.no_grad():
        plain.weight[module.token_ids] = module.rows

    model.embedding.token_embedding = plain
    return len(module.token_ids)


def apply_lora(model, r: int = 8, alpha: int = 16, target_modules=ATTENTION_PROJECTIONS):
    """Freeze the whole model, then wrap the named attention projections in
    every block with a LoRALinear.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced = 0
    for block in model.blocks:
        for name in target_modules:
            linear = getattr(block.attn, name)
            lora_linear = LoRALinear(base=linear, r=r, alpha=alpha)

            setattr(block.attn, name, lora_linear)
            replaced += 1

    return replaced

def merge_lora(model, target_modules=ATTENTION_PROJECTIONS):
    """Fold every LoRALinear's detour back into an ordinary nn.Linear.
    """

    merged = 0
    for block in model.blocks:
        for name in target_modules:
            module = getattr(block.attn, name)
            if not isinstance(module, LoRALinear):
                continue

            plain = nn.Linear(
                module.base.in_features,
                module.base.out_features,
                bias=module.base.bias is not None
            ).to(module.base.weight.device)

            with torch.no_grad():
                plain.weight.copy_(module.merged_weight())
                if module.base.bias is not None:
                    plain.bias.copy_(module.base.bias)

            setattr(block.attn, name, plain)
            merged += 1

    return merged

