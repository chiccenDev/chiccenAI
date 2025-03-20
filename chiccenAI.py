from openai import OpenAI
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import discord

#os.chdir(r"C:\Users\Josh\Desktop\programming\AI\DeepSeek\chiccenAI")

with open ("./token.txt" , "r") as f:
    token = f.read()

# load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine-tuned-model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

client = OpenAI(api_key="sk-8d911eacf0d544048227bd7be67c4121", base_url="https://api.deepseek.com")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user in message.mentions:
        response = generate_response(message.content, model, tokenizer)
        await message.channel.send(response)


def generate_response(prompt, model, tokenizer, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response
    outputs = model.generate(
        inputs,
        do_sample=True,
        max_length=max_length,  # Maximum length of the generated response
        num_return_sequences=1,  # Number of responses to generate
        no_repeat_ngram_size=2,  # Prevent repetition of n-grams
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Nucleus sampling
        temperature=0.2,  # Sampling temperature
    )

    # Decode the generated tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

client.run(token)