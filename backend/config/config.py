from pydantic import BaseSettings


class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    OPENAI_ORG: str
    OPENAI_API_KEY: str
    SERPAPI_API_KEY: str

    class Config:
        env_file = "../.env"


settings = Settings()
