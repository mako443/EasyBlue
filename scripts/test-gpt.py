from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_organization="org-WneNpGKlIbUZfwIgdehAfJ6u",
    openai_api_key="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",
    temperature=0.0,
)

system_message = SystemMessage(
    content="""
    You are a helpful assistant that reads in a list of history sentences and an input sentence. 
    You will then say if the input sentence is roughly similar to any of the history sentences.
    You will answer in the format yes: (name of history sentence) or no.
    """
)


human_message = HumanMessage(
    content="The history sentences are 'I like to eat apples' and 'I like to eat bananas'. The input sentence is 'I hate apples'."
)
result = chat([system_message, human_message])
print(result)

human_message = HumanMessage(content="The input sentence is 'I love bananas'.")
result = chat([human_message])
print(result)
