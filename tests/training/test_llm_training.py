import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def train_simple_llm():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
