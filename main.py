from typing import List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
import json
from generate_graph import CombinedGraphCreator
from confidentialgraph import ConfidentialSubgraphGenerator
from generateanswer import QueryProcessor

app = FastAPI()

class AuthModel(BaseModel):
    token: str

class KnowledgeData(BaseModel):
    summary: str
    content: str
    authors: List[str]

class QueryData(BaseModel):
    query: str

@app.post("/add_knowledge")
async def add_knowledge(data: KnowledgeData, token: AuthModel = Depends()):
    with open("./PacemakerInnovationsData/authentication.json", "r") as f:
        auth_data = json.load(f)
    user_data = next((user for user in auth_data if user["token"] == token.token), None)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid token")

    authors = [user_data["nameSurname"]]
    start_date = datetime(2022, 1, 1)
    current_date = datetime.now()
    quarter_index = (current_date.year - start_date.year) * 4 + (current_date.month - 1) // 3

    json_data = {
        "date": current_date.strftime("%Y-%m-%d"),
        "quarter_index": quarter_index,
        "summary": data.summary,
        "authors": authors,
        "content": data.content
    }

    with open("./PacemakerInnovationsData/test_node_ingestion.json", "w") as f:
        json.dump(json_data, f)

    adder = CombinedGraphCreator(json_file="./PacemakerInnovationsData/test_node_ingestion.json", output_file="./PacemakerInnovationsData/graph.pt")
    adder.add_node()

    return {"message": "Knowledge added successfully"}

@app.post("/query")
async def query_knowledge(data: QueryData, token: AuthModel = Depends()):
    with open("./PacemakerInnovationsData/authentication.json", "r") as f:
        auth_data = json.load(f)
    user_data = next((user for user in auth_data if user["token"] == token.token), None)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid token")

    generator = ConfidentialSubgraphGenerator(graph_file="./PacemakerInnovationsData/graph.pt")
    generator.generate_subgraph(user_token=token.token)

    processor = QueryProcessor(graph_file=f"./cache/{token.token}.pt")
    response = processor.perform_query(data.query)

    return {"answer": response}