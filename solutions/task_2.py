from datasets import load_dataset

# Downloading only a subset of training and validattion data for speeding up the process
raw_datasets = load_dataset("wmt15", "fr-en", split=['train[:100000]', 'validation[:1000]'])

print(raw_datasets)

# Train
print(raw_datasets[0]["translation"][:5])

# Validation
print(raw_datasets[1]["translation"][:5])