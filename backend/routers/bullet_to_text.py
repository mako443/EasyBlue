from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.config.config import settings


class BulletData(BaseModel):
    bullet_points: List[str]


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)
router = APIRouter()


@router.post("/bullet_to_text")
def bullet_to_text(data: BulletData):
    bullet_points = data.bullet_points
    text = "\n".join(bullet_points)

    message = HumanMessage(content=f"Convert the following bullet points into an extensive, continuous text: {text}")
    result = chat([message])
    return {"result": result.content}
