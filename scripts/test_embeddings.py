import pickle
import numpy as np
from dotenv import load_dotenv
import os

from langchain.embeddings import OpenAIEmbeddings

load_dotenv()


def get_topk(embeddings, query, k=5):
    similarities = np.dot(embeddings, query)
    indices = np.argsort(-1.0 * similarities)
    # distances = np.linalg.norm(embeddings - query, axis=1)
    # indices = np.argsort(distances)

    indices = indices[:k]
    return indices


def get_closest_lines(embeddings, lines, query, api):
    query = query.strip().replace("\n", "")
    query = np.array(api.embed_query(query)).flatten()
    indices = get_topk(embeddings, query, k=5)
    return [lines[i] for i in indices]


with open("data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    embeddings = data["embeddings"]
    lines = data["lines"]
    assert len(lines) == len(embeddings)

api = OpenAIEmbeddings(openai_organization=os.getenv("OPENAI_ORG"), openai_api_key=os.getenv("OPENAI_API_KEY"))

get_closest_lines(embeddings, lines, "found damage", api)
