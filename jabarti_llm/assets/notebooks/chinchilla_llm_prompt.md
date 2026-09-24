## Task: Calculate the Chinchilla Token-to-Parameter Ratio

I’m planning to train a decoder-only Transformer language model using the following configuration:

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    d_ff: int | None = None
    max_seq_len: int = 1024
    dropout: float = 0.1
    n_layers: int = 12

    n_heads: int = 12
    qkv_bias: bool = False
    pad_id: int = 0

    tie_weights: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )

    @property
    def d_k(self):
        """The width of one head's Query, Key and Value vectors."""
        return self.d_model // self.n_heads

    @classmethod
    def jabarti(cls):
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=8,
            dropout=0.05
        )

    @classmethod
    def tiny(cls):
        return cls(
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=256
        )
```

I will be using:

```python
ModelConfig.jabarti()
```

Therefore, the effective configuration is approximately:

```text
vocab_size   = 32,000
d_model      = 512
d_ff         = 2,048
n_layers     = 8
n_heads      = 8
max_seq_len  = 1,024
tie_weights  = True
qkv_bias     = False
```

My dataset contains:

```text
Training:
729,562,657 tokens
7,053,893 documents

Evaluation:
11,728,528 tokens
114,259 documents
```

### Questions

1. Estimate the total number of trainable parameters in the `jabarti` model.

   Please show the calculation and break it down into approximately:

   * token embeddings
   * attention parameters
   * MLP/feed-forward parameters
   * LayerNorms and other relevant parameters
   * output/language-model head

   Take `tie_weights=True` into account, so the output projection should share weights with the token embedding matrix.

   If any architectural detail required for an exact parameter count is missing, clearly state your assumptions.

2. Calculate the **training tokens per parameter**:

   ```text
   Chinchilla ratio = number of training tokens / number of model parameters
   ```

   Use the **729,562,657 training tokens** for this calculation. Do not include the evaluation set unless there is a specific reason to do so.

3. Compare my resulting tokens-per-parameter ratio with the commonly cited Chinchilla compute-optimal guideline of roughly **20 training tokens per parameter**.

4. Based on that comparison, tell me:

   * whether this model is approximately under-trained, compute-optimal, or over-trained relative to the Chinchilla guideline;
   * approximately how many training tokens would correspond to a 20 tokens/parameter target for this model;
   * how many epochs over my 729.6M-token training dataset that would represent.

5. Also calculate the inverse perspective:

   * given my fixed dataset size of **729,562,657 tokens**, approximately what model parameter count would correspond to the 20 tokens/parameter Chinchilla guideline?

Please show the formulas and intermediate calculations rather than only giving the final numbers.

Also distinguish between:

* **unique dataset tokens**
* **total tokens seen during training across multiple epochs**

because I may train for more than one epoch.
