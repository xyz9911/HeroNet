import math
from model import HeroGAN
from utils import InputExample, compute_metrics, tokenizer
from training_args_gan import GANTrainingArguments
from trainer_gan import GANTrainer
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

batch_size = 64


def get_samples():
    train_samples = []
    dev_samples = []
    candidate_pool = []

    with open('datasets/advising_st2/train_demo.json', 'r') as f:
        data = json.load(f)
        for item in data:
            train_samples.append(InputExample(texts=[item['context'], item['truth'], item['neg'], item['es']]))

    with open('datasets/advising_st2/dev.json', 'r') as f:
        data = json.load(f)
        for item in data:
            dev_samples.append(InputExample(texts=[item['context'], item['truth'], item['neg'], item['es']]))

    with open('datasets/advising_st2/candidate_pool.json', 'r') as f:
        data = json.load(f)
        for item in data:
            candidate_pool.append(item['utterance'])

    return train_samples, dev_samples, candidate_pool


train_samples, dev_samples, candidate_pool = get_samples()

model = HeroGAN.from_pretrained('T5')

model.config.max_length = 32

max_input_length = 256
max_target_length = 128

args = GANTrainingArguments(
    "Checkpoints",
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    predict_with_generate=True,
    save_total_limit=20,
    save_strategy="epoch"
)

warmup_steps = math.ceil(len(train_samples) * 5 * 0.1)

trainer = GANTrainer(
    model,
    max_input_length,
    warmup_steps,
    args,
    train_samples=train_samples,
    eval_samples=dev_samples,
    candidate_pool=candidate_pool,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
