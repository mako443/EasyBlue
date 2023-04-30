from fastapi import APIRouter
from pydantic import BaseModel
import pickle

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from backend.config.config import settings


class SelectData(BaseModel):
    text: str


def translate(text, from_language, to_language):
    message = HumanMessage(
        content=f"Translate the following text from {from_language.capitalize()} to {to_language.capitalize()}: {text}"
    )
    result = chat([message])
    return result.content


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)
router = APIRouter()

with open("../data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    lines = data["lines"]
lines = [line.strip().lower() for line in lines]


@router.post("/select_suggestions")
def select_suggestions(data: SelectData, language: str = "english"):
    search_terms = data.text
    if language != "english":
        search_terms = translate(search_terms, language, "english")
    search_terms = search_terms.strip().lower().split()
    found_lines = []
    for line in lines:
        if all([term in line.lower() for term in search_terms]):
            found_lines.append(line)
        if len(found_lines) >= 5:
            break

    if language != "english":
        for i in range(len(found_lines)):
            found_lines[i] = translate(found_lines[i], "english", language)

    return {
        "results": found_lines,
    }
