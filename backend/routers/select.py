from fastapi import APIRouter
from pydantic import BaseModel
import pickle


class SelectData(BaseModel):
    text: str


router = APIRouter()

with open("../data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    lines = data["lines"]
lines = [line.strip().lower() for line in lines]


@router.post("/select_suggestions")
def select_suggestions(data: SelectData):
    search_terms = data.text.strip().lower().split()
    found_lines = []
    for line in lines:
        if all([term in line for term in search_terms]):
            found_lines.append(line)
        if len(found_lines) >= 5:
            break

    return {
        "results": found_lines,
    }
