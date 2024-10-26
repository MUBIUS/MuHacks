import sys
import os
from config import settings
from database.analyzer import EnhancedDatabaseAnalyzer
from services.assistant import LlamaSQLAssistant
from services.cache import QueryCache
from services.metrics import PerformanceMetrics
from models.schemas import QueryRequest
from models.schemas import QueryResponse

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Initialize services
db_analyzer = EnhancedDatabaseAnalyzer(settings.DATABASE_URL)
assistant = LlamaSQLAssistant(settings.MODEL_PATH, db_analyzer)
cache = QueryCache(settings.REDIS_URL)
metrics = PerformanceMetrics(settings.DATABASE_URL)

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    api_key: str = Depends(API_KEY_HEADER)
):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        cached_result = cache.get_cached_result(request.query)
        if cached_result:
            return QueryResponse.parse_raw(cached_result)
        
        sql = assistant.generate_sql(request.query)
        results = assistant.execute_query(sql)
        
        response = QueryResponse(
            sql=sql,
            results=results.to_dict(orient='records'),
            execution_time=0.0,  # Add actual timing
            column_names=list(results.columns),
            row_count=len(results)
        )
        
        cache.cache_result(request.query, response.json())
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
