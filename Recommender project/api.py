from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from ml_backend import final_recommendation


app = FastAPI(title="Laptop Recommendation API")


# ----------- Allow frontend (JS) to call this API -----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for hackathon; restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------- Request schema -----------

class QueryRequest(BaseModel):
    query: str
    top_n: int = 3


# ----------- Health check -----------

@app.get("/")
def root():
    return {"status": "API is running"}


# ----------- Main endpoint -----------

@app.post("/recommend")
def recommend(req: QueryRequest):

    result = final_recommendation(
        query=req.query,
        top_n=req.top_n
    )

    return {
        "results": result
    }
