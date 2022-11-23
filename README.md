# HeroNet
Codes are publicly available for dasfaa2023 manuscript -> 

### HeroNet: A Hybrid Retrieval-Generation Network for Conversational Bots

## Requirements
absl-py==1.0.0
datasets==1.18.3
huggingface-hub==0.2.1
nltk
rouge-score==0.0.4
sacrebleu==1.5.1
sentencepiece==0.1.95
tokenizers==0.10.3
torch==1.12.1
transformers==4.12.3

## Usage
The hyperparameters for the HeroNet can be found in ''training_args_gan.py''.
You can adjust them in the GANTrainingArguments class from the ''TrainGAN.py''.

If you want to run the model, use the command:
```shell
python TrainGAN.py
```

## Dataset
We only provide datasets for demonstration, to access the full dataset, please refer to the paper [A Large-Scale Corpus for Conversation Disentanglement](https://aclanthology.org/P19-1374/)

## Output
The output of our model is presented in the ''result'' directory