import evaluate

chrf = evaluate.load("chrf")

predictions = [
    "This is an examplary sentence.",
    "Try different sentences for the reference and observe the change in scores."
]
references = [
    ["This is an examplary sentence."],
    ["Try different sentences for the reference and observe the change in scores."]
]

results = chrf.compute(predictions=predictions, references=references)

print(round(results["score"], 1))