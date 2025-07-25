from dataclasses import dataclass, field

@dataclass
class SparseArguments:
    # sparse arguments
    sparse_score_accumulation_examples: int = field(default=1024)
    sparse_mask_update_steps: int = field(default=128)
    sparse_enable: bool = field(default=False)
    sparse_method: str = field(default='magnitude')
    sparse_k: float = field(default=0.02)
    sparse_global_topk: bool = field(default=False)
    sparse_alpha: float = field(default=1)
    sparse_dropout: float = field(default=0.0)

    # lora arguments
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    # lora_bias: str = field(default="none")
    pissa_enable: bool = field(default=False)