import requests
import json

# Replace with your actual API key
API_KEY = "AIzaSyCvZQe0QQP8UNk5-GIBaUZbialV-0CuX7Q"

# API endpoint for Gemma 3 27B (text generation)
API_URL = "https://generativelanguage.googleapis.com/v1beta3/models/gemma-3b-it:generateText?key=" + API_KEY

# Your prompt
prompt = "Write a short story about a cat who goes on an adventure."

# Request payload
payload = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ],
    "generation_config": {
        "temperature": 0.7,  # Adjust for creativity (0.0 - 1.0)
        "top_p": 0.9,        # Adjust for diversity (0.0 - 1.0)
        "max_output_tokens": 256, # Maximum length of the generated text
    },
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUAL",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
}

# Make the API request
try:
    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    response.raise_for_status()  # Raise an exception for bad status codes

    # Parse the response
    json_response = response.json()

    # Extract the generated text
    generated_text = json_response["candidates"][0]["content"]

    # Print the generated text
    print(generated_text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except (KeyError, IndexError) as e:
    print(f"Error parsing API response: {e}")
    print(f"Response content: {response.content}") #Print the full response for debugging