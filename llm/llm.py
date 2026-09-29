from groq import Groq

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqLLM:
    def __init__(self, api_key=None, model=DEFAULT_MODEL):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        return response.choices[0].message.content