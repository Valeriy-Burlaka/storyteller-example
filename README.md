# Storyteller

## Description

The Storyteller project is an experimental Python application that converts images into short audio stories. It leverages OpenAI's GPT-3 and Hugging Face's transformers library to perform tasks like image-to-text conversion, text-based story generation, and text-to-speech conversion.

## Installation and Setup Steps

### Prerequisites

- Python 3.x
- Conda (Optional but recommended for environment isolation)

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/storyteller.git
    cd storyteller
    ```

2. **Setup the Environment**

     - Using Conda:

       ```bash
       conda env create -f environment.yaml
       conda activate <env-name>
       ```

3. **Setup Environment Variables**

    - Copy `.env.example` to `.env`

        ```bash
        cp .env.example .env
        ```

    - Edit the `.env` file to add your API keys and tokens.

4. **Run the App**

    ```bash
    python app.py
    ```

### What to Expect

After running `app.py`, a Streamlit application will start, and you should see a web interface with an option to upload an image. Upon uploading an image, the application will generate a short story based on the image's content and convert the story to audio. You can view the generated scenario, read the story, and listen to the audio story directly in the web interface.
