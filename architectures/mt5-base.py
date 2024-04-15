from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).bfloat16()