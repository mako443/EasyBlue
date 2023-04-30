"""Path hack to enable normal uvicorn start and reliable imports.
"""
import sys

sys.path.append("../")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import select, autocomplete2, translate, bullet_to_text, pdf_create, transcribe

app = FastAPI()
origins = ["*", "*.paypal.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(select.router)
app.include_router(autocomplete2.router)
app.include_router(translate.router)
app.include_router(bullet_to_text.router)
app.include_router(pdf_create.router)
app.include_router(transcribe.router)

app.mount("/files/", StaticFiles(directory="../files/"), name="files")
