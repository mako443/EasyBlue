from typing import List
from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pydf
from fastapi.templating import Jinja2Templates
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import numpy as np

from backend.config.config import settings


class PDFData(BaseModel):
    bullet_points: str


def translate_bullets(text, from_language, to_language):
    message = HumanMessage(
        content=f"Translate the following bullet points from {from_language.capitalize()} to {to_language.capitalize()} and only return the translated bullet points: {text}"
    )
    result = chat([message])
    return result.content


def translate_text(text, from_language, to_language):
    message = HumanMessage(
        content=f"Translate the following text from {from_language.capitalize()} to {to_language.capitalize()} and only return the translated text: {text}"
    )
    result = chat([message])
    return result.content


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)


router = APIRouter()

templates = Jinja2Templates(directory="../data/")


@router.post("/create_pdf")
def create_pdf(data: PDFData, language: str = "english"):
    bullet_points = data.bullet_points

    if language != "english":
        bullet_points = translate_bullets(bullet_points, language, "english")
        print("Translated to english:", bullet_points, "END \n")

    message = HumanMessage(
        content=f"Convert the following bullet points into an extensive, continuous text and return only that text: {bullet_points}"
    )
    full_text = chat([message]).content

    message = HumanMessage(
        content=f"Summarize the following text into a short description of one sentence and only return that sentence: {full_text}"
    )
    summary = chat([message]).content

    bullet_points = bullet_points.split(". ")
    all_bullet_points = []
    for bullet_point in bullet_points:
        all_bullet_points.extend(bullet_point.split("\n"))

    print("ALL BULLETS:", all_bullet_points)

    t = templates.get_template("html_template.html")
    html = t.render({"description": summary, "solution": full_text, "bullet_points": all_bullet_points})
    pdf = pydf.generate_pdf(html)

    filename = str(np.random.randint(100000)) + ".pdf"
    with open(f"../files/{filename}", "wb") as f:
        f.write(pdf)

    return {"url": f"/files/{filename}"}
