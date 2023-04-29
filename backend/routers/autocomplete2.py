from fastapi import APIRouter
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import PromptTemplate
import pickle

from backend.config.config import settings


class AutocompleteData(BaseModel):
    text: str


with open("../data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    lines = data["lines"]


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)


router = APIRouter()
with open("../data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    embeddings = data["embeddings"]
    lines = data["lines"]
    assert len(lines) == len(embeddings)


@router.post("/autocomplate_suggestions")
def autocomplate_suggestions(data: AutocompleteData):
    template = """You are a writing assistant for technicians. Based on the start of their text, generate an autocompletion. 
                    Only finish the sentence. Keep it to one sentence. Do not generate additional text beyond the description.
                    Examples: {examples}
                    Current Input: {userinput}"""

    promptTemplate = PromptTemplate(input_variables=["userinput", "examples"], template=template)

    examples = "\n".join(lines)
    prompt = promptTemplate.format(userinput=data.text, examples=examples)

    result = chat([HumanMessage(content=prompt)]).content
    if ". " in result:
        result = result.split(". ")[0] + ". "
    return {
        "result": result,
    }
