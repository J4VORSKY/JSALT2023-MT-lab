import evaluate

comet = evaluate.load("comet")

predictions = [
    "This is an examplary sentence.",
    "Try different sentences for the reference and observe the change in scores."
]
references = [
    ["This is an examplary sentence."],
    ["Try different sentences for the reference and observe the change in scores."]
]

results = comet.compute(predictions=predictions, references=references)

print(round(results["score"], 1))