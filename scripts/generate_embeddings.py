import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
import pickle
import numpy as np

load_dotenv()

with open("data/pdfs.txt", "r") as f:
    lines = f.readlines()

# Preprocess lines
lines = [line.strip().replace("\n", "") for line in lines]
lines = [line for line in lines if len(line) > 5]

split_lines = []
for line in lines:
    split_lines.extend(line.split(". "))
lines = split_lines

for i in range(len(lines)):
    lines[i] = lines[i].strip()
    lines[i] = lines[i] + "." if not line.endswith(".") else lines[i]

api = OpenAIEmbeddings(openai_organization=os.getenv("OPENAI_ORG"), openai_api_key=os.getenv("OPENAI_API_KEY"))

embeds = np.array(api.embed_documents(lines))

with open("data/embeddings.pkl", "wb") as f:
    data = {
        "lines": lines,
        "embeddings": embeds,
    }
    pickle.dump(data, f)
