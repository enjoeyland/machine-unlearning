from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)