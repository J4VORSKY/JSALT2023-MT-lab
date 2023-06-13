from datasets import load_dataset

# Downloading only a subset of training and validattion data for speeding up the process
raw_datasets = load_dataset("wmt15", "fr-en", split=['train[:100000]', 'validation[:1000]'])

print(raw_datasets)
print(raw_datasets["train"][:5])