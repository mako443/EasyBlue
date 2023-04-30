from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.config.config import settings


class BulletData(BaseModel):
    bullet_points: List[str]


def translate(text, from_language, to_language):
    message = HumanMessage(
        content=f"Translate the following text from {from_language.capitalize()} to {to_language.capitalize()} and only return the translated text: {text}"
    )
    result = chat([message])
    return result.content


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)
router = APIRouter()


@router.post("/bullet_to_text")
def bullet_to_text(data: BulletData, language: str = "english"):
    bullet_points = data.bullet_points
    text = "\n".join(bullet_points)

    if language != "english":
        bullet_points = translate(bullet_points, language, "english")
        print("Translated to english:", bullet_points, "END \n")

    message = HumanMessage(content=f"Convert the following bullet points into an extensive, continuous text: {text}")
    result = chat([message]).content

    if language != "english":
        result = translate(result, "english", language)
        print("Translated to", language, ":", result, "END \n")

    return {"result": result}
