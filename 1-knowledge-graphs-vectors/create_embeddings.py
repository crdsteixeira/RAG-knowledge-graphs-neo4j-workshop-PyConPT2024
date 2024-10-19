import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = llm.embeddings.create(
        input="A mysterious spaceship lands Earth",
        model="text-embedding-ada-002"
    )

print(response.data[0].embedding)
