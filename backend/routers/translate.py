from fastapi import APIRouter
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.config.config import settings


class TranslateDate(BaseModel):
    from_language: str
    to_language: str
    text: str


router = APIRouter()


@router.post("/translate_text")
def translate_text(data: TranslateDate):
    return {"input_text": data.text, "from_language": data.from_language}
