import os

from dotenv import find_dotenv, load_dotenv
import openai
import requests
import streamlit as st
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

    print("generate_story_openai: Starting API request...")

    api_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages,
    )

    print("generate_story_openai: OpenAI API response:", api_response)
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


def generate_story(scenario: str) -> str:
    story = None

    if USE_OPENAI_API:
        story = generate_story_openai(scenario)
    else:
        story = generate_story_oss_model(scenario)

    return story


def generate_speech(text: str, output_name="audio.flac") -> str:
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_HUB_API_TOKEN}",
    }
    payload = {
        "inputs": text,
    }

    print("generate_speech: Starting API request...")

    response = requests.post(API_URL, headers=headers, json=payload)
    with open(output_name, "wb") as f:
        f.write(response.content)

    return output_name


def old_main():
    text_on_image = image_to_text("Brick_sign_large__compressed.png")
    if USE_OPENAI_API:
        story = generate_story_openai(text_on_image)
    else:
        story = generate_story_oss_model(text_on_image)
    print("Story:", story)
    generate_speech(story)


def main():
    st.set_page_config(page_title="Storyteller", page_icon="📖")

    st.header("Turn any image into a compelling audio story")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        print("Uploaded file:", uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as image_file:
            image_file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded image.",
                 use_column_width=True)

        scenario = image_to_text(uploaded_file.name)
        story = generate_story(scenario)
        audio_file_name = generate_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio(audio_file_name)


if __name__ == "__main__":
    main()
