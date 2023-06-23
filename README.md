# JSALT 2023 - Machine translation lab

This is the description of Machine translation lab exercises during the summer school at JSALT, Le Mans. This document should guide you to get familiar with the Hugging Face library in the context of machine translation. Specifically, it provides several tasks such as downloading models, data, evaluation metrics, and using them to train models from scratch, finetune models and evaluate them with different metrics.

Each task consists of a description and a solution which is hidden at the beggining. You can first try it on your own and look at the solution later. All the python scripts are also included in the `solutions` directory.

## Setting the environment

### Task 0

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

### Task 1

Now, create a python script which downloads a model of your choice from the `Helsinki-NLP` family from [here](https://huggingface.co/Helsinki-NLP). Then, use the model and translate an examplary sentence and print its translation on the standard output.

Note that the `Helsinki-NLP` family is a collection of models trained in the framework of [Marian NMT](https://marian-nmt.github.io/), an efficient NMT implementation written in pure C++. The models have been converted to pyTorch using the transformers library by Hugging Face. You can find more details about `MarianMT` [here](https://huggingface.co/docs/transformers/model_doc/marian).

In this tutorial, we will use the model `en-fr`.

<details>
<summary>Solution</summary>

```python
from transformers import MarianMTModel, MarianTokenizer

src_text = [
    "This is a sentence in English that we want to translate to French."
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

### Task 2

The next step is to download and prepare the data for training. For example, use one of the `wmtXX` datasets and a language pair of your choice. Specifically, download the dataset (load only first 100k training and 1000 validation samples to speed up the process), get familiar with its structure and print first 5 items.

<details>
<summary>Solution</summary>

```python
from datasets import load_dataset

# Downloading only a subset of training and validattion data for speeding up the process
raw_datasets = load_dataset("wmt15", "fr-en", split=['train[:100000]', 'validation[:1000]'])

print(raw_datasets)

# Train
print(raw_datasets[0]["translation"][:5])

# Validation
print(raw_datasets[1]["translation"][:5])
```

Output:
```bash
[Dataset({
    features: ['translation'],
    num_rows: 100000
}), Dataset({
    features: ['translation'],
    num_rows: 1000
})]
[{'en': 'Resumption of the session', 'fr': 'Reprise de la session'}, {'en': 'I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.', 'fr': 'Je déclare reprise la session du Parlement européen qui avait été interrompue le vendredi 17 décembre dernier et je vous renouvelle tous mes vux en espérant que vous avez passé de bonnes vacances.'}, {'en': "Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful.", 'fr': 'Comme vous avez pu le constater, le grand "bogue de l\'an 2000" ne s\'est pas produit. En revanche, les citoyens d\'un certain nombre de nos pays ont été victimes de catastrophes naturelles qui ont vraiment été terribles.'}, {'en': 'You have requested a debate on this subject in the course of the next few days, during this part-session.', 'fr': 'Vous avez souhaité un débat à ce sujet dans les prochains jours, au cours de cette période de session.'}, {'en': "In the meantime, I should like to observe a minute' s silence, as a number of Members have requested, on behalf of all the victims concerned, particularly those of the terrible storms, in the various countries of the European Union.", 'fr': "En attendant, je souhaiterais, comme un certain nombre de collègues me l'ont demandé, que nous observions une minute de silence pour toutes les victimes, des tempêtes notamment, dans les différents pays de l'Union européenne qui ont été touchés."}]
[{'en': 'Sounds like a typical rugby club to me.', 'fr': "Ça m'a l'air d'être un club de rugby typique."}, {'en': 'At an English university, perhaps...', 'fr': 'Dans une université anglaise, peut-être...'}, {'en': 'Not like any rugby club I know about in NZ.', 'fr': 'Rien à voir avec les clubs de rugby que je connais en NZ.'}, {'en': "It doesn't make it all right though, does it?", 'fr': 'Mais ça ne justifie rien, si ?'}, {'en': "Of course it's not right, but the original premise that this is rife in rugby is just pertinent bollix...", 'fr': "Bien sûr que non, mais la prémisse qui dit que c'est courant dans le rugby est du gros n'importe quoi..."}]
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
import evaluate

sacrebleu = evaluate.load("sacrebleu")

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

Combine solutions from previous tasks to train a model. You can use a `Trainer`, see [here](https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt).

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
model_name = "Helsinki-NLP/opus-mt-en-fr"
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

# compute_metrics inspired by https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Uncomment this if you want to print predictions
    # print("Prediction:", decoded_preds[0])
    # print("Reference:", decoded_labels[0])
    
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
        model_target = tokenizer([e["fr"] for e in examples["translation"]], truncation=True)
    model_inputs["labels"] = model_target["input_ids"]

    return model_inputs

raw_datasets = load_dataset("wmt15", "fr-en", split=['train[:100000]', 'validation[:100]'])

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
    learning_rate=1e-3,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
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
    compute_metrics=compute_metrics
)

trainer.train()
```

Output:
```bash
{'loss': 6.6711, 'learning_rate': 0.0002, 'epoch': 0.06}                                                                                
{'eval_loss': 6.28677225112915, 'eval_bleu': 0.1204, 'eval_gen_len': 55.92, 'eval_runtime': 23.5412, 'eval_samples_per_second': 4.248, 'eval_steps_per_second': 0.17, 'epoch': 0.06}                                                                                            
{'loss': 4.6937, 'learning_rate': 0.0004, 'epoch': 0.13}                                                                                
{'eval_loss': 5.674839973449707, 'eval_bleu': 0.4148, 'eval_gen_len': 59.14, 'eval_runtime': 22.497, 'eval_samples_per_second': 4.445, 'eval_steps_per_second': 0.178, 'epoch': 0.13}                                                                                           
{'loss': 4.1794, 'learning_rate': 0.0006, 'epoch': 0.19}                                                                                
{'eval_loss': 5.4645538330078125, 'eval_bleu': 0.6928, 'eval_gen_len': 56.79, 'eval_runtime': 18.1481, 'eval_samples_per_second': 5.51, 'eval_steps_per_second': 0.22, 'epoch': 0.19}                                                                                           
{'loss': 3.865, 'learning_rate': 0.0008, 'epoch': 0.26}                                                                                 
{'eval_loss': 5.306863784790039, 'eval_bleu': 1.6476, 'eval_gen_len': 27.96, 'eval_runtime': 4.4549, 'eval_samples_per_second': 22.447, 'eval_steps_per_second': 0.898, 'epoch': 0.26}                                                                                          
{'loss': 3.6928, 'learning_rate': 0.001, 'epoch': 0.32}                                                                                 
{'eval_loss': 5.251842498779297, 'eval_bleu': 2.0643, 'eval_gen_len': 35.6, 'eval_runtime': 5.0805, 'eval_samples_per_second': 19.683, 'eval_steps_per_second': 0.787, 'epoch': 0.32}                                                                                           
{'loss': 3.5901, 'learning_rate': 0.0009058380414312618, 'epoch': 0.38}                                                                 
{'eval_loss': 5.087807655334473, 'eval_bleu': 1.2203, 'eval_gen_len': 42.67, 'eval_runtime': 12.312, 'eval_samples_per_second': 8.122, 'eval_steps_per_second': 0.325, 'epoch': 0.38}                                                                                           
{'loss': 3.3773, 'learning_rate': 0.0008116760828625236, 'epoch': 0.45}                                                                 
{'eval_loss': 4.878002643585205, 'eval_bleu': 2.626, 'eval_gen_len': 30.68, 'eval_runtime': 4.4822, 'eval_samples_per_second': 22.31, 'eval_steps_per_second': 0.892, 'epoch': 0.45}                                                                                            
{'loss': 3.223, 'learning_rate': 0.0007175141242937854, 'epoch': 0.51}                                                                  
{'eval_loss': 4.801403999328613, 'eval_bleu': 1.9422, 'eval_gen_len': 29.38, 'eval_runtime': 4.9496, 'eval_samples_per_second': 20.204, 'eval_steps_per_second': 0.808, 'epoch': 0.51}                                                                                          
{'loss': 3.0709, 'learning_rate': 0.000623352165725047, 'epoch': 0.58}                                                                  
{'eval_loss': 4.66360330581665, 'eval_bleu': 4.0011, 'eval_gen_len': 28.04, 'eval_runtime': 5.2486, 'eval_samples_per_second': 19.053, 'eval_steps_per_second': 0.762, 'epoch': 0.58}                                                                                           
{'loss': 2.9161, 'learning_rate': 0.0005291902071563088, 'epoch': 0.64}                                                                 
{'eval_loss': 4.574631214141846, 'eval_bleu': 5.5121, 'eval_gen_len': 28.51, 'eval_runtime': 4.2558, 'eval_samples_per_second': 23.497, 'eval_steps_per_second': 0.94, 'epoch': 0.64}                                                                                           
{'loss': 2.7881, 'learning_rate': 0.0004350282485875706, 'epoch': 0.7}                                                                  
{'eval_loss': 4.405792236328125, 'eval_bleu': 4.2598, 'eval_gen_len': 33.04, 'eval_runtime': 6.1833, 'eval_samples_per_second': 16.173, 'eval_steps_per_second': 0.647, 'epoch': 0.7}                                                                                           
{'loss': 2.6581, 'learning_rate': 0.0003408662900188324, 'epoch': 0.77}                                                                 
{'eval_loss': 4.312582492828369, 'eval_bleu': 4.6802, 'eval_gen_len': 37.67, 'eval_runtime': 7.2567, 'eval_samples_per_second': 13.78, 'eval_steps_per_second': 0.551, 'epoch': 0.77}                                                                                           
{'loss': 2.5477, 'learning_rate': 0.00024670433145009414, 'epoch': 0.83}                                                                
{'eval_loss': 4.134029388427734, 'eval_bleu': 7.1339, 'eval_gen_len': 29.1, 'eval_runtime': 4.2295, 'eval_samples_per_second': 23.644, 'eval_steps_per_second': 0.946, 'epoch': 0.83}                                                                                           
{'loss': 2.4373, 'learning_rate': 0.00015254237288135595, 'epoch': 0.9}                                                                 
{'eval_loss': 4.065229415893555, 'eval_bleu': 7.5816, 'eval_gen_len': 29.22, 'eval_runtime': 4.349, 'eval_samples_per_second': 22.994, 'eval_steps_per_second': 0.92, 'epoch': 0.9}                                                                                             
{'loss': 2.345, 'learning_rate': 5.83804143126177e-05, 'epoch': 0.96}                                                                   
{'eval_loss': 3.9800426959991455, 'eval_bleu': 8.4406, 'eval_gen_len': 30.89, 'eval_runtime': 4.4788, 'eval_samples_per_second': 22.328, 'eval_steps_per_second': 0.893, 'epoch': 0.96}                                                                                         
{'train_runtime': 520.9683, 'train_samples_per_second': 191.95, 'train_steps_per_second': 2.998, 'train_loss': 3.423431220646857, 'epoch': 1.0}
```
</details>

### Task 7 (Finetuning)

Remove the reinitialization part from the previous script and only finetune the model. Note that you might want to decrease the learning rate.

<details>
<summary>Solution</summary>

```python
args = Seq2SeqTrainingArguments(
    f"models/{model_name.split('/')[1]}.finetuned",
    evaluation_strategy = "steps",
    logging_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True
)
```

Output:

```bash
{'loss': 1.3192, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.06}                                                                 
{'eval_loss': 1.5500624179840088, 'eval_bleu': 29.5866, 'eval_gen_len': 27.86, 'eval_runtime': 6.485, 'eval_samples_per_second': 15.42, 'eval_steps_per_second': 0.617, 'epoch': 0.06}                                                                                            
{'loss': 1.2494, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.13}                                                                  
{'eval_loss': 1.5322158336639404, 'eval_bleu': 30.2329, 'eval_gen_len': 27.61, 'eval_runtime': 6.4892, 'eval_samples_per_second': 15.41, 'eval_steps_per_second': 0.616, 'epoch': 0.13}                                                                                           
{'loss': 1.2334, 'learning_rate': 6e-06, 'epoch': 0.19}                                                                                  
{'eval_loss': 1.5338757038116455, 'eval_bleu': 29.7931, 'eval_gen_len': 27.51, 'eval_runtime': 6.4111, 'eval_samples_per_second': 15.598, 'eval_steps_per_second': 0.624, 'epoch': 0.19}                                                                                          
{'loss': 1.2217, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.26}                                                                  
{'eval_loss': 1.5355517864227295, 'eval_bleu': 29.6001, 'eval_gen_len': 27.68, 'eval_runtime': 6.407, 'eval_samples_per_second': 15.608, 'eval_steps_per_second': 0.624, 'epoch': 0.26}                                                                                           
{'loss': 1.205, 'learning_rate': 1e-05, 'epoch': 0.32}                                                                                   
{'eval_loss': 1.5382646322250366, 'eval_bleu': 30.1429, 'eval_gen_len': 27.62, 'eval_runtime': 6.4171, 'eval_samples_per_second': 15.583, 'eval_steps_per_second': 0.623, 'epoch': 0.32}                                                                                          
{'loss': 1.2075, 'learning_rate': 9.058380414312619e-06, 'epoch': 0.38}                                                                  
{'eval_loss': 1.5464560985565186, 'eval_bleu': 30.1443, 'eval_gen_len': 27.77, 'eval_runtime': 6.4813, 'eval_samples_per_second': 15.429, 'eval_steps_per_second': 0.617, 'epoch': 0.38}                                                                                          
{'loss': 1.1922, 'learning_rate': 8.116760828625236e-06, 'epoch': 0.45}                                                                  
{'eval_loss': 1.5470266342163086, 'eval_bleu': 29.8518, 'eval_gen_len': 27.61, 'eval_runtime': 6.3876, 'eval_samples_per_second': 15.655, 'eval_steps_per_second': 0.626, 'epoch': 0.45}                                                                                          
{'loss': 1.1896, 'learning_rate': 7.175141242937854e-06, 'epoch': 0.51}                                                                  
{'eval_loss': 1.5515811443328857, 'eval_bleu': 29.0653, 'eval_gen_len': 27.69, 'eval_runtime': 6.4642, 'eval_samples_per_second': 15.47, 'eval_steps_per_second': 0.619, 'epoch': 0.51}                                                                                           
{'loss': 1.1937, 'learning_rate': 6.233521657250471e-06, 'epoch': 0.58}                                                                  
{'eval_loss': 1.5522204637527466, 'eval_bleu': 29.4558, 'eval_gen_len': 27.86, 'eval_runtime': 6.5, 'eval_samples_per_second': 15.385, 'eval_steps_per_second': 0.615, 'epoch': 0.58}
{'loss': 1.1784, 'learning_rate': 5.2919020715630885e-06, 'epoch': 0.64}                                                                 
{'eval_loss': 1.5516700744628906, 'eval_bleu': 29.6477, 'eval_gen_len': 27.7, 'eval_runtime': 6.473, 'eval_samples_per_second': 15.449, 'eval_steps_per_second': 0.618, 'epoch': 0.64}                                                                                            
{'loss': 1.1765, 'learning_rate': 4.350282485875706e-06, 'epoch': 0.7}                                                                   
{'eval_loss': 1.5537399053573608, 'eval_bleu': 29.6423, 'eval_gen_len': 27.88, 'eval_runtime': 6.4312, 'eval_samples_per_second': 15.549, 'eval_steps_per_second': 0.622, 'epoch': 0.7}                                                                                           
{'loss': 1.1788, 'learning_rate': 3.4086629001883244e-06, 'epoch': 0.77}                                                                 
{'eval_loss': 1.5528733730316162, 'eval_bleu': 29.7681, 'eval_gen_len': 27.56, 'eval_runtime': 6.302, 'eval_samples_per_second': 15.868, 'eval_steps_per_second': 0.635, 'epoch': 0.77}                                                                                           
{'loss': 1.187, 'learning_rate': 2.4670433145009417e-06, 'epoch': 0.83}                                                                  
{'eval_loss': 1.5520755052566528, 'eval_bleu': 29.7988, 'eval_gen_len': 27.86, 'eval_runtime': 6.4759, 'eval_samples_per_second': 15.442, 'eval_steps_per_second': 0.618, 'epoch': 0.83}                                                                                          
{'loss': 1.1774, 'learning_rate': 1.5254237288135596e-06, 'epoch': 0.9}                                                                  
{'eval_loss': 1.5530320405960083, 'eval_bleu': 29.9099, 'eval_gen_len': 27.81, 'eval_runtime': 6.4859, 'eval_samples_per_second': 15.418, 'eval_steps_per_second': 0.617, 'epoch': 0.9}                                                                                           
{'loss': 1.1875, 'learning_rate': 5.83804143126177e-07, 'epoch': 0.96}                                                                   
{'eval_loss': 1.5529881715774536, 'eval_bleu': 29.8751, 'eval_gen_len': 27.71, 'eval_runtime': 6.4282, 'eval_samples_per_second': 15.557, 'eval_steps_per_second': 0.622, 'epoch': 0.96}                                                                                          
{'train_runtime': 1138.3078, 'train_samples_per_second': 87.85, 'train_steps_per_second': 1.372, 'train_loss': 1.2049754749644885, 'epoch': 1.0}
```

</details>