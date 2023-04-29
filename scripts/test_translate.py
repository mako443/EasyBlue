import os
from dotenv import load_dotenv
import pickle
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()

chat = ChatOpenAI(
    temperature=0, openai_organization=os.getenv("OPENAI_ORG"), openai_api_key=os.getenv("OPENAI_API_KEY")
)


def translate(from_language, to_language, text):
    message = HumanMessage(
        content=f"Translate the following text from {from_language.capitalize()} to {to_language.capitalize()}: {text}"
    )
    result = chat([message])
    return result.content


res = translate("english", "spanish", "I am a cat")
