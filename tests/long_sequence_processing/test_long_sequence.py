import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_long_sequence_processing():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    
    # Generate a long input sequence
    long_input = "Hello" * 1000
    inputs = tokenizer(long_input, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    outputs = model(**inputs)