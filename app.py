import os

from dotenv import find_dotenv, load_dotenv
import openai
from transformers import pipeline

USE_OPENAI_API = True

load_dotenv(find_dotenv())
if USE_OPENAI_API:
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def image_to_text(image_url: str) -> str:
    # More details about tasks supported by the "transformers" library can be found here:
    #   https://huggingface.co/tasks
    # Also:
    #   https://huggingface.co/tasks/image-to-text

    # Have a bit of an issue here:
    # 1. Doesn't work with `model="Salesforce/blip-image-captioning-large"` parameter:
    # """
    # Traceback (most recent call last):
    #   File "/Users/val/Documents/Edu/AI/custom-langchain-example/app.py", line 31, in <module>
    #     image_to_text("./Brick_sign_large__compressed.png")
    #   File "/Users/val/Documents/Edu/AI/custom-langchain-example/app.py", line 14, in image_to_text
    #     img2text = pipeline(
    #   File "/Users/val/anaconda3/envs/llm-playground/lib/python3.10/site-packages/transformers/pipelines/__init__.py", line 648, in pipeline
    #     config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **model_kwargs)
    #   File "/Users/val/anaconda3/envs/llm-playground/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 796, in from_pretrained
    #     config_class = CONFIG_MAPPING[config_dict["model_type"]]
    #   File "/Users/val/anaconda3/envs/llm-playground/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 503, in __getitem__
    #     raise KeyError(key)
    # KeyError: 'blip'
    # """
    #
    # 2. Works without specifying the model, but warned that it's not a good approach:
    # """
    # (llm-playground) Moo-Book:custom-langchain-example val$ python3 app.py
    # No model was supplied, defaulted to ydshieh/vit-gpt2-coco-en and revision 65636df (https://huggingface.co/ydshieh/vit-gpt2-coco-en).
    # Using a pipeline without specifying a model name and revision in production is not recommended.
    # /Users/val/anaconda3/envs/llm-playground/lib/python3.10/site-packages/transformers/generation_utils.py:1359: UserWarning: Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to 20 (`self.config.max_length`). Controlling `max_length` via the config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
    #   warnings.warn(
    # image-to-text response: [{'generated_text': 'a stop sign with a red and white arrow '}]
    # generated text: a stop sign with a red and white arrow
    # """
    #
    # Still nice to get at least some results
    #
    # 3. Works with no issues using another model - `model="nlpconnect/vit-gpt2-image-captioning"`:
    # """
    # val$ python3 app.py
    # ...
    # Downloading (…)lve/main/config.json: 100%|███████████████████████████████████████████████████████████████| 4.61k/4.61k [00:00<00:00, 4.56MB/s]
    # Downloading pytorch_model.bin: 100%|███████████████████████████████████████████████████████████████████████| 982M/982M [00:56<00:00, 17.5MB/s]
    # Downloading (…)okenizer_config.json: 100%|███████████████████████████████████████████████████████████████████| 241/241 [00:00<00:00, 1.72MB/s]
    # Downloading (…)olve/main/vocab.json: 100%|█████████████████████████████████████████████████████████████████| 798k/798k [00:00<00:00, 1.79MB/s]
    # Downloading (…)olve/main/merges.txt: 100%|█████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 1.38MB/s]
    # Downloading (…)/main/tokenizer.json: 100%|███████████████████████████████████████████████████████████████| 1.36M/1.36M [00:00<00:00, 5.48MB/s]
    # Downloading (…)cial_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████| 120/120 [00:00<00:00, 897kB/s]
    # Downloading (…)rocessor_config.json: 100%|████████████████████████████████████████████████████████████████████| 228/228 [00:00<00:00, 847kB/s]
    # /Users/val/anaconda3/envs/llm-playground/lib/python3.10/site-packages/transformers/generation_utils.py:1359: UserWarning: Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to 20 (`self.config.max_length`). Controlling `max_length` via the config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
    #   warnings.warn(
    # image-to-text response: [{'generated_text': 'a stop sign with a red and white arrow '}]
    # generated text: a stop sign with a red and white arrow
    # """
    #
    # 4. Oh, and I couldn't make it work using the "Load model directly" example from the model's page:
    # """
    # # Load model directly
    # from transformers import AutoProcessor, AutoModelForSeq2SeqLM

    # processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/blip-image-captioning-large")
    # """
    #
    # val$ python3 app.py
    # """
    # Traceback (most recent call last):
    #   File "/Users/val/Documents/Edu/AI/custom-langchain-example/app.py", line 7, in <module>
    #     processor = AutoProcessor.from_pretrained(
    #   File "/Users/val/anaconda3/envs/llm-playground/lib/python3.10/site-packages/transformers/models/auto/processing_auto.py", line 248, in from_pretrained
    #     return processor_class.from_pretrained(
    # AttributeError: 'NoneType' object has no attribute 'from_pretrained'
    # """

    pipe = pipeline("image-to-text",
                    model="nlpconnect/vit-gpt2-image-captioning")
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

    print("generate_story: OpenAI API response:", api_response)
    response_message = api_response["choices"][0]["message"]

    return response_message


def generate_story_oss_model(scenario: str) -> str:
    return "TODO: Implement integration "

# text to speech


text_on_image = image_to_text("Brick_sign_large__compressed.png")
if USE_OPENAI_API:
    story = generate_story_openai(text_on_image)
else:
    story = generate_story_oss_model(text_on_image)
print("Story:", story)
