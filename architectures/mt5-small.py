from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)