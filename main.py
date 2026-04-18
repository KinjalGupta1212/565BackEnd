from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from Rag_Pipeline import DataAnnotationRAG
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class Query(BaseModel):
    text: str
    attributes: list[str]

class Response(BaseModel):
    response: str

async def lifespan(app: FastAPI):
    app.state.agent = DataAnnotationRAG()
    yield


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:5173"] #frontend url

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#will have to add an argument to the 
@app.post("/chat")
async def get_answer(query: Query, request: Request):
    agent = request.app.state.agent
    response = await agent.get_response(query.text, query.attributes)
    return Response(response=response)
    