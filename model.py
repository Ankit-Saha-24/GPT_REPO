import json
import os
import pandas as pd
from google import genai
from google.genai import types

class MizoTranslator:
    def __init__(self, api_key, model_dir="mizo_translation_model"):
        self.client = genai.Client(api_key=api_key)
        
        # Load prompt instruction
        prompt_path = os.path.join(model_dir, 'system_instruction.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_instruction = f.read()
            
    def predict(self, english_text):
        response = self.client.models.generate_content(
            model='gemini-3.6-flash',
            contents=english_text,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.3
            )
        )
        return response.text.strip()

# Usage Example:
# translator = MizoTranslator(api_key="YOUR_GEMINI_API_KEY")
# print(translator.predict("How are you today?"))
