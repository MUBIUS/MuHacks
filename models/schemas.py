from pydantic import BaseModel, Field
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query to convert to SQL")
    max_results: int = Field(1000, description="Maximum number of results to return")

class QueryResponse(BaseModel):
    sql: str
    results: List[Dict[str, Any]]
    execution_time: float
    column_names: List[str]
    row_count: int