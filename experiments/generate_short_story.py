import os

from dotenv import find_dotenv, load_dotenv
import openai

load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")

SCENARIO = "araffe sitting in a chair with a bear on his lap"


def use_davinci():
    prompt = """
  You are a story teller.
  You can tell a short story based on a simple narrative.
  The story should be no longer than 50 words and a few sentences.

  CONTEXT: {scenario}
  STORY:
  """.format(scenario=SCENARIO)
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        max_tokens=60,
    )
    print("DaVinci response", response, "\n\n")


def use_gpt_turbo_with_system_setup():
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
            "content": prompt_template.format(scenario=SCENARIO),
        },
    ]

    api_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages,
    )

    print('GPT Turbo response (With system setup):', api_response, "\n\n")
    response_message = api_response["choices"][0]["message"]

    return response_message["content"]


def use_gpt_turbo_with_user_message_only():
    prompt_template = """
  You are a story teller. You can tell a short story based on a simple narrative.
  The story should be no longer than 25 words and a few sentences.

  CONTEXT: {scenario}
  STORY:
  """
    gpt_messages = [
        {
            "role": "user",
            "content": prompt_template.format(scenario=SCENARIO),
        },
    ]

    api_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages,
    )

    print('GPT Turbo response (With user prompt only):', api_response, "\n\n")
    response_message = api_response["choices"][0]["message"]

    return response_message["content"]


if __name__ == "__main__":
    use_davinci()
    use_gpt_turbo_with_system_setup()
    use_gpt_turbo_with_user_message_only()
