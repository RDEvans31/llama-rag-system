import json
from dotenv import load_dotenv
from fastapi import  FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from uuid import uuid4
from typing import Any
import io
import os
import time
import tempfile

logger = logging.getLogger("uvicorn")

app = FastAPI(title="HuggingFace LLM Wrapper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello_world():
    
    logger.info(f"Custom LLM Api running!")

    return {
        "status": 200,
        "body": "Api running."
    }

@app.generate("/generate/")
def generate()


