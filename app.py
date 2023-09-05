import os

from dotenv import find_dotenv, load_dotenv
import openai
import requests
from transformers import pipeline

USE_OPENAI_API = True

load_dotenv(find_dotenv())
HUGGINGFACE_HUB_API_TOKEN = os.environ.get("HUGGINGFACE_HUB_API_TOKEN")

if USE_OPENAI_API:
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def image_to_text(image_url: str) -> str:
    # More details about tasks supported by the "transformers" library can be found here:
    #   https://huggingface.co/tasks
    # Also:
    #   https://huggingface.co/tasks/image-to-text

    pipe = pipeline("image-to-text",
                    # model="nlpconnect/vit-gpt2-image-captioning")
                    model="Salesforce/blip-image-captioning-large")
    model_response = pipe(image_url)

    print("image-to-text response:", model_response)
    text_on_image = model_response[0]["generated_text"]

    print("generated text:", text_on_image)

    return text_on_image


def generate_story_openai(scenario: str) -> str:
    prompt_template = """
    CONTEXT: {scenario}
    STORY:
    """
    gpt_messages = [
        {
            "role": "system",
            "content": """
            You are a story teller. You can tell a short story based on a simple narrative.
            The story should be no longer than 25 words and a few sentences.
            """,
        },
        {
            "role": "user",
            "content": prompt_template.format(scenario=scenario),
        },
    ]
    api_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages,
    )

    # print("generate_story: OpenAI API response:", api_response)
    response_message = api_response["choices"][0]["message"]

    return response_message["content"]


def generate_story_oss_model(scenario: str) -> str:
    prompt_template = """
    You are a story teller.
    You can tell a short story based on a simple narrative.
    The story should be no longer than 25 words and a few sentences.

    CONTEXT: {scenario}
    STORY:
    """
    pipe = pipeline("text-generation", model="TheBloke/Llama-2-70B-GPTQ")  # Needs GPU :(
    model_response = pipe(prompt_template.format(scenario=scenario))

    # print("text-generation response:", model_response)

    return "TODO: Implement integration (wasn't trivial to pick a model)"


def generate_speech(text: str) -> str:
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_HUB_API_TOKEN}",
    }
    payload = {
        "inputs": text,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open("audio.flac", "wb") as f:
        f.write(response.content)


text_on_image = image_to_text("Brick_sign_large__compressed.png")
if USE_OPENAI_API:
    story = generate_story_openai(text_on_image)
else:
    story = generate_story_oss_model(text_on_image)
print("Story:", story)

speech = generate_speech(story)
