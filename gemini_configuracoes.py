from google import genai

client = genai.Client(api_key="GEMIMI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-pro", contents="Explica o que é machine learning."
)
print(response.text)