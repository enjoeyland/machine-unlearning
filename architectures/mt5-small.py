from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)