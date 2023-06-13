from transformers import (
    DataCollatorWithPadding,
    MarianMTModel,
    MarianTokenizer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    logging,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer
    )
from datasets import load_dataset
import evaluate

import torch
import numpy as np

# Model and metric loading
model_name = "Helsinki-NLP/opus-mt-en-cs"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
metric     = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# Preprocessing logits for effective memory usage
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = torch.argmax(logits, axis=-1)
    return preds, labels

# compute_metrics inspired by https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print("Prediction:", decoded_preds[0])
    print("Reference:", decoded_labels[0])
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
   
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
   
    return result

# Dataset preparation
def tokenize_function(examples):
    model_inputs = tokenizer([e["en"] for e in examples["translation"]], truncation=True)
    with tokenizer.as_target_tokenizer():
        model_target = tokenizer([e["cs"] for e in examples["translation"]], truncation=True)
    model_inputs["labels"] = model_target["input_ids"]

    return model_inputs

raw_datasets = load_dataset("wmt19", "cs-en", split=['train[:100000]', 'validation[:1000]'])

tokenized_train = raw_datasets[0].map(tokenize_function, batched=True)
tokenized_valid = raw_datasets[1].map(tokenize_function, batched=True)
data_collator   = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
args = Seq2SeqTrainingArguments(
    f"models/{model_name.split('/')[1]}.from-scratch",
    evaluation_strategy = "steps",
    logging_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True
)

# Training
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

trainer.train()