import sys
import time
from openai import OpenAI


client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")

prompt = "Describe the 5 most important elements in the EU AI Act using max 50 words for each."
print(f"Sending prompt: '{prompt}'...")

start_time = time.time()
response = client.chat.completions.create(
    model=sys.argv[1],
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)
end_time = time.time()

duration = end_time - start_time
output_text = response.choices[0].message.content
tokens_generated = len(output_text.split()) # Estimation very rough OK

print(f"Got result:\n{output_text}")
print("================")
print(f"Inference duration: {duration:.2f} seconds.")
print(f"Average throughput: {tokens_generated / duration:.2f} tokens/s.")

