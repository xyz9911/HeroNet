from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from dataclasses import dataclass, field


@dataclass
class GANTrainingArguments(Seq2SeqTrainingArguments):
    t5_pre_epochs: int = field(default=10)
    discriminator_pre_epochs: int = field(default=10)
    adv_epochs: int = field(default=20)
    adv_generator_steps: int = field(default=1)
    adv_discriminator_steps: int = field(default=1)
    t5_pre_lr: float = field(default=4e-4)
    st5_pre_lr: float = field(default=1e-4)
    t5_lr: float = field(default=2e-4)
    st5_lr: float = field(default=1e-4)
    margin: float = field(default=0.5)
    margin_es: float = field(default=0.5)
    train_k: int = field(default=4)
    train_k_es: int = field(default=2)
    eval_k: int = field(default=50)
    eval_k_es: int = field(default=50)
    eval_sample_beams: int = field(default=1)
    use_euclidean_distance: bool = field(default=True)
    es_truth_ratio: float = field(default=0.3)
