"""Path hack to enable normal uvicorn start and reliable imports.
"""
import sys

sys.path.append("../")

from fastapi import FastAPI

from backend.routers import select, autocomplete, translate

app = FastAPI()

# Routers
app.include_router(select.router)
app.include_router(autocomplete.router)
app.include_router(translate.router)
