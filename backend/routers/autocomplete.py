from fastapi import APIRouter
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import pickle

from backend.config.config import settings


def get_topk(embeddings, query, k=5):
    similarities = np.dot(embeddings, query)
    indices = np.argsort(-1.0 * similarities)
    indices = indices[:k]
    return indices


def get_closest_lines(embeddings, lines, query, api):
    query = query.strip().replace("\n", "")
    query = np.array(api.embed_query(query)).flatten()
    indices = get_topk(embeddings, query, k=5)
    return [lines[i] for i in indices]


class AutocompleteData(BaseModel):
    text: str


api = OpenAIEmbeddings(openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)
router = APIRouter()
with open("../data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    embeddings = data["embeddings"]
    lines = data["lines"]
    assert len(lines) == len(embeddings)


@router.post("/autocomplate_suggestions")
def autocomplate_suggestions(data: AutocompleteData):
    closest_lines = get_closest_lines(embeddings, lines, data.text, api)

    return {
        "suggestions": closest_lines,
    }
