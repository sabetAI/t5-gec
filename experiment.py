# Mask filling only works for bart-large
from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
TXT = "old [comment]."

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
input_ids = tokenizer([TXT], return_tensors='pt', padding="max_length")['input_ids']
print(input_ids)
logits = model(input_ids)

print(logits)
