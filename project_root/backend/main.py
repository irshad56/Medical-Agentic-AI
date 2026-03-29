from fastapi import FastAPI
from backend.routes.api import router

app = FastAPI(title="Agentic AI System")

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Backend is running"}