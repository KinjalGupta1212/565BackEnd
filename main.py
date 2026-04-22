from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from Rag_Pipeline import DataAnnotationRAG
from pydantic import BaseModel, model_serializer, model_validator
from fastapi.middleware.cors import CORSMiddleware

class Query(BaseModel):
    text: str
    attributes: list[str]

class AttributeInsight(BaseModel):
    questions: list[str]
    similar_comments: list[str]
    disagreeing_comments: dict[str, dict[str, int]]


class ChatAnnotationResponse(BaseModel):
    """
    Matches the dict returned by DataAnnotationRAG.get_response on success:
    one nested object per requested attribute, plus table_info and targeted_subgroups.
    """

    table_info: dict[str, dict[str, list[str]]]
    targeted_subgroups: list[str]
    attributes: dict[str, AttributeInsight]

    @model_validator(mode="before")
    @classmethod
    def split_top_level_attribute_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        reserved = frozenset({"table_info", "targeted_subgroups"})
        attrs: dict[str, Any] = {}
        core: dict[str, Any] = {}
        for key, value in data.items():
            if key in reserved:
                core[key] = value
            else:
                attrs[key] = value
        core["attributes"] = attrs
        return core

    @model_serializer(mode="plain")
    def flatten_for_json(self) -> dict[str, Any]:
        return {
            **self.attributes,
            "table_info": self.table_info,
            "targeted_subgroups": self.targeted_subgroups,
        }

async def lifespan(app: FastAPI):
    app.state.agent = DataAnnotationRAG()
    yield


app = FastAPI(lifespan=lifespan)

origins = ["*"] #frontend url

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#will have to add an argument to the 
@app.post("/chat", response_model=None)
async def get_answer(query: Query, request: Request):
    agent = request.app.state.agent
    raw = await agent.get_response(query.text, query.attributes)
    if isinstance(raw, str):
        return JSONResponse({"error": raw}, status_code=503)
    return ChatAnnotationResponse.model_validate(raw).model_dump()
    