import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode("Anurag Rana")
print(integers)

output_str=""
for i in integers:
    output_str += f"{i} -> {tokenizer.decode([i])}\n"
    
print(output_str)

print(tokenizer.decode(integers))