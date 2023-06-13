import evaluate

comet = evaluate.load("comet")


source = [
    "Dem Feuer konnte Einhalt geboten werden",
    "Schulen und Kindergärten wurden eröffnet.",
]
hypothesis = [
    "The fire could be stopped",
    "Schools and kindergartens were open",
]
reference = [
    "They were able to control the fire.",
    "Schools and kindergartens opened",
]
comet_score = comet.compute(
    predictions=hypothesis, references=reference, sources=source
)
print(comet_score)
