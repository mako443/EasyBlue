from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pydf
from fastapi.templating import Jinja2Templates
import numpy as np


class PDFData(BaseModel):
    description: str
    solution: str
    notes: str


router = APIRouter()

templates = Jinja2Templates(directory="../data/")


@router.post("/create_pdf")
def create_pdf(data: PDFData):
    t = templates.get_template("html_template.html")
    html = t.render({"description": data.description, "solution": data.solution, "notes": data.notes})
    pdf = pydf.generate_pdf(html)

    filename = str(np.random.randint(100000)) + ".pdf"
    with open(f"../files/{filename}", "wb") as f:
        f.write(pdf)

    return {"url": f"/files/{filename}"}
