from dotenv import load_dotenv
load_dotenv()
import os

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

def call_gemini(prompt):
  interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=prompt
  )
  return interaction.output_text
  