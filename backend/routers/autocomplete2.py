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


def translate(text, from_language, to_language):
    message = HumanMessage(
        content=f"Translate the following text from {from_language.capitalize()} to {to_language.capitalize()} and only return the translated text: {text}"
    )
    result = chat([message])
    return result.content


chat = ChatOpenAI(temperature=0, openai_organization=settings.OPENAI_ORG, openai_api_key=settings.OPENAI_API_KEY)


router = APIRouter()
with open("../data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    embeddings = data["embeddings"]
    lines = data["lines"]
    assert len(lines) == len(embeddings)


@router.post("/autocomplete_suggestions")
def autocomplete_suggestions(data: AutocompleteData, language: str = "english"):
    # template = """You are a writing assistant for technicians. Based on the start of their text, generate an autocompletion.
    #                 Only finish the sentence. Keep it to one sentence. Do not generate additional text beyond the description.
    #                 Examples: {examples}
    #                 Current Input: {userinput}"""

    template = """You are a writing assistant for technicians. Based on the start of their text, generate an autocompletion. 
                    Only return the complete sentence. Keep it to one sentence. Do not generate additional text beyond the description.
                    Examples: {examples}
                    Current Input: {userinput}"""

    user_input = data.text
    print("user_input:", user_input)
    if language != "english":
        user_input = translate(user_input, language, "english")
        print("Translated to english:", user_input, "END")

    promptTemplate = PromptTemplate(input_variables=["userinput", "examples"], template=template)

    examples = "\n".join(lines)
    prompt = promptTemplate.format(userinput=user_input, examples=examples)

    result = chat([HumanMessage(content=prompt)]).content
    if ". " in result:
        result = result.split(". ")[0] + ". "

    if language != "english":
        result = translate(result, "english", language)
        print("Translated to", language, ":", result, "END")
    return {
        "result": result,
    }
