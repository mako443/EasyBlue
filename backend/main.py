"""Path hack to enable normal uvicorn start and reliable imports.
"""
import sys

sys.path.append("../")

from fastapi import FastAPI

from backend.routers import select, autocomplete2, translate, bullet_to_text, pdf_create

app = FastAPI()

# Routers
app.include_router(select.router)
app.include_router(autocomplete2.router)
app.include_router(translate.router)
app.include_router(bullet_to_text.router)
app.include_router(pdf_create.router)
