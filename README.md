# JSALT 2023 - Machine translation lab

This is the description of Machine translation lab exercises during the summer school at JSALT, Le Mans. This document should guide you to get familiar with the Hugging Face library in the context of machine translation. Specifically, it provides several tasks such as downloading models, data, evaluation metrics, and using them to train models from scratch, finetune models and evaluate them with different metrics.

Each task consists of a description and a solution which is hidden at the beggining. You can first try it on your own and look at the solution later. All the python scripts are also included in the `solutions` directory.

## Setting the environment

### Task 1

You first task is to create a python virtual environment and install `torch`, `transformers`, `sentencepiece`, `datasets`, `evaluate`, `sacremoses`, `sacrebleu`, `unbabel-comet`, `accelerate`.

<details>
<summary>Solution</summary>

```bash
/path/to/python -m venv name-of-your-env
```

```bash
source name-of-your-env/bin/activate
```

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install torch transformers sentencepiece datasets evaluate sacremoses sacrebleu unbabel-comet accelerate
```

</details>

## Models

### Task 2

Now, create a python script which downloads a model of your choice from the `Helsinki-NLP` family from [here](https://huggingface.co/Helsinki-NLP). Then, use the model and translate an examplary sentence and print its translation on the standard output.

Note that the `Helsinki-NLP` family is a collection of models trained in the framework of [Marian NMT](https://marian-nmt.github.io/), an efficient NMT implementation written in pure C++. The models have been converted to pyTorch using the transformers library by Hugging Face. You can find more details about `MarianMT` [here](https://huggingface.co/docs/transformers/model_doc/marian).

In this tutorial, we will use the model `en-fr`.

<details>
<summary>Solution</summary>

```python
from transformers import MarianMTModel, MarianTokenizer

src_text = [
    "This is a sentence in english that we want to translate to french."
]

model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model     = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
```

Output:
```bash
["C'est une phrase en anglais que nous voulons traduire en français."]
```

</details>

## Data

The next step is to download and prepare the data for training. For example, use one of the `wmtXX` datasets and a language pair of your choice. Specifically, download the dataset (load only first 100k training and 1000 validation samples to speed up the process), get familiar with its structure and print first 5 items.

<details>
<summary>Solution</summary>

```python
from datasets import load_dataset

# Downloading only a subset of training and validattion data for speeding up the process
raw_datasets = load_dataset("wmt15", "fr-en", split=['train[:100000]', 'validation[:1000]'])

print(raw_datasets)
print(raw_datasets["train"][:5])
```

Output:
```bash
DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 7270695
    })
    validation: Dataset({
        features: ['translation'],
        num_rows: 2983
    })
})
{'translation': [{'cs': 'Následný postup na základě usnesení Parlamentu: viz zápis', 'en': "Action taken on Parliament's resolutions: see Minutes"}, {'cs': 'Předložení dokumentů: viz zápis', 'en': 'Documents received: see Minutes'}, {'cs': 'Písemná prohlášení (článek 116 jednacího řádu): viz zápis', 'en': 'Written statements (Rule 116): see Minutes'}, {'cs': 'Texty smluv dodané Radou: viz zápis', 'en': 'Texts of agreements forwarded by the Council: see Minutes'}, {'cs': 'Složení Parlamentu: viz zápis', 'en': 'Membership of Parliament: see Minutes'}]}
```

</details>

## Evaluation

Another important part of training is its evaluation. For this, we use three metrics: `BLEU`, `ChrF` and `COMET`.

In the following tasks, examine how to use these metrics on their Hugging Face documentation pages.

### Task 3 ([sacrebleu](https://huggingface.co/spaces/evaluate-metric/sacrebleu))

Read more about sacrebleu [here](https://github.com/mjpost/sacrebleu): <em>"SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores. Inspired by Rico Sennrich's multi-bleu-detok.perl, it produces the official WMT scores but works with plain text. It also knows all the standard test sets and handles downloading, processing, and tokenization for you."</em>

<details>
<summary>Solution</summary>

```python
from datasets import load_metric

sacrebleu = load_metric("sacrebleu")

predictions = [
    "This is an examplary sentence.",
    "Try different sentences for the reference and observe the change in scores."
]
references = [
    ["This is an examplary sentence."],
    ["Try different sentences for the reference and observe the change in scores."]
]

results = sacrebleu.compute(predictions=predictions, references=references)

print(round(results["score"], 1))
```

Output:
```bash
100.0
```

</details>

### Task 4 ([ChrF](https://huggingface.co/spaces/evaluate-metric/chrf))

ChrF and ChrF++ are two MT evaluation metrics. They both use the F-score statistic for character n-gram matches, and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment. We use the implementation that is already present in sacrebleu.

<details>
<summary>Solution</summary>

Use `metric = load_metric("chrf")` instead of `metric = load_metric("sacrebleu")`.

</details>

### Task 5 ([COMET](https://huggingface.co/spaces/evaluate-metric/comet))

<em>"COMET (Crosslingual Optimized Metric for Evaluation of Translation) is a new neural framework for training multilingual machine translation (MT) evaluation models. COMET is designed to predict human judgments of MT quality (such as MQM scores). The resulting metric can be used to automate the process of evaluation, improving efficiency and speed."</em>

NOTE: Since COMET is currently (temporarily?) unavailable on Hugging Face, try to practice it using [command line arguments](https://github.com/Unbabel/COMET).

## Training

Now, we compare training models from scratch and finetuning a model. In particular, choose a model (e.g. `Helsinki-NLP/opus-mt-en-fr`) and use it to
1. train it from scratch, and
2. finetune it.

### Task 6 (Training from scratch)

Combine solutions from previous tasks to train a model. You can use a `Trainer`, see [here](https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt). Do not forget to reinitialize model weights, for example by:

```python
for layer in model.model.encoder.layers + model.model.decoder.layers:
    layer.apply(model._init_weights)
```

<details>
<summary>Solution</summary>

```python
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

# Reinitializing model weights
for layer in model.model.encoder.layers + model.model.decoder.layers:
    layer.apply(model._init_weights)

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

raw_datasets = load_dataset("wmt19", "cs-en", split=['train[:100000]', 'validation[:1000]'],
    cache_dir="/home/javorsky/personal_work_troja/.cache/huggingface/datasets/")

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
```
</details>

### Task 7 (Finetuning)

Remove the reinitialization part from the previous script. We use ten times smaller learning rate:
- `learning_rate=0.00002`
