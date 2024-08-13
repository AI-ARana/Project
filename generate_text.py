from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch  # Make sure to import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=100, temperature=1, top_k=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Create an attention mask
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example usage
prompt = "Shoolini University in India"
generated_text = generate_text(prompt)
print(generated_text)

'''Generate text based on different types of prompts to see how context affects the output.


# Different prompts
prompts = [
    "In a galaxy far, far away",
    "Once upon a time in a land of magic",
    "The future of technology is",
]

# Generate text for each prompt
for prompt in prompts:
    print(f"Prompt: {prompt}")
    print("Generated text:")
    print(generate_text(prompt))
    print("-" * 40)

'''

'''Allow the user to input their own prompt and generate text based on that input.

# User input
user_prompt = input("Enter your prompt: ")
generated_text = generate_text(user_prompt)
print("Generated text:")
print(generated_text)


'''
'''Post-process the generated text to remove unwanted characters or format it in a specific way.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate and post-process text
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process the generated text
    text = text.strip()  # Remove leading and trailing whitespace
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    return text

# Example usage
prompt = "In the future,"
generated_text = generate_text(prompt)
print("Generated text with post-processing:")
print(generated_text)

'''