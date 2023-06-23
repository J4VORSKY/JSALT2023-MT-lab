from transformers import MarianMTModel, MarianTokenizer

src_text = [
    "This is a sentence in English that we want to translate to French."
]

model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model     = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])