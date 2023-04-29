from fastapi import APIRouter
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from backend.config.config import settings


class TranslateDate(BaseModel):
    from_language: str
    to_language: str
    text: str


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)
router = APIRouter()


@router.post("/translate_text")
def translate_text(data: TranslateDate):
    message = HumanMessage(
        content=f"Translate the following text from {data.from_language.capitalize()} to {data.to_language.capitalize()}: {data.text}"
    )
    result = chat([message])
    return {"result": result.content}
