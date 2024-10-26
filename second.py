# from llama_cpp import Llama
# import sqlite3
# from typing import List, Dict, Tuple, Optional, Any
# import json
# import logging
# import pandas as pd
# import time
# from datetime import datetime
# import hashlib
# import asyncio
# from fastapi import FastAPI, HTTPException, WebSocket, Depends, Query
# from fastapi.security import APIKeyHeader
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel, Field
# import sqlparse
# import numpy as np
# from rich.console import Console
# from rich.table import Table
# from rich.syntax import Syntax
# from rich.progress import Progress
# import plotly.express as px
# import plotly.graph_objects as go
# from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Float, Integer, ForeignKey
# from sqlalchemy.orm import sessionmaker
# import redis
# import jwt
# from functools import lru_cache
# import uvicorn
# from database.analyzer import DatabaseAnalyzer
# from services.assistant import LlamaSQLAssistant

# # Logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('sql_assistant.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class QueryCache:
#     def __init__(self, redis_url: str = "redis://localhost:6379"):
#         self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
#         self.ttl = 3600  # Cache TTL in seconds

#     def get_cache_key(self, query: str) -> str:
#         return f"sql_query:{hashlib.md5(query.encode()).hexdigest()}"

#     def get_cached_result(self, query: str) -> Optional[str]:
#         return self.redis_client.get(self.get_cache_key(query))

#     def cache_result(self, query: str, result: str):
#         self.redis_client.setex(self.get_cache_key(query), self.ttl, result)

# class QueryValidator:
#     @staticmethod
#     def validate_syntax(sql: str) -> Tuple[bool, str]:
#         try:
#             parsed = sqlparse.parse(sql)
#             if not parsed:
#                 return False, "Empty query"
            
#             # Check basic SQL structure
#             statement = parsed[0]
#             if not statement.get_type():
#                 return False, "Invalid SQL statement type"

#             return True, "Valid SQL syntax"
#         except Exception as e:
#             return False, f"Syntax error: {str(e)}"

#     @staticmethod
#     def check_security(sql: str) -> Tuple[bool, str]:
#         sql_lower = sql.lower()
        
#         # Check for dangerous operations
#         dangerous_keywords = ['drop', 'truncate', 'delete', 'update', 'insert', 'alter', 'create']
#         for keyword in dangerous_keywords:
#             if keyword in sql_lower:
#                 return False, f"Unauthorized {keyword.upper()} operation detected"

#         return True, "Query passed security check"

# class QueryOptimizer:
#     @staticmethod
#     def optimize_query(sql: str) -> str:
#         parsed = sqlparse.parse(sql)[0]
        
#         # Basic query optimizations
#         optimized = sql
        
#         # Replace SELECT * with specific columns
#         if 'select *' in optimized.lower():
#             logger.warning("Consider specifying columns instead of using SELECT *")
        
#         # Add LIMIT if not present
#         if 'limit' not in optimized.lower():
#             optimized += " LIMIT 1000"
        
#         return optimized

# class PerformanceMetrics:
#     def __init__(self):
#         self.metrics_db = create_engine('sqlite:///metrics.db')
#         self.init_metrics_table()

#     def init_metrics_table(self):
#         metadata = MetaData()
#         self.metrics_table = Table(
#             'query_metrics', metadata,
#             Column('id', Integer, primary_key=True),
#             Column('query_hash', String),
#             Column('execution_time', Float),
#             Column('timestamp', DateTime),
#             Column('success', Integer),
#             Column('error_message', String, nullable=True)
#         )
#         metadata.create_all(self.metrics_db)

#     def record_metric(self, query: str, execution_time: float, success: bool, error_message: Optional[str] = None):
#         query_hash = hashlib.md5(query.encode()).hexdigest()
#         with self.metrics_db.connect() as conn:
#             conn.execute(
#                 self.metrics_table.insert().values(
#                     query_hash=query_hash,
#                     execution_time=execution_time,
#                     timestamp=datetime.now(),
#                     success=int(success),
#                     error_message=error_message
#                 )
#             )

#     def get_performance_report(self) -> Dict[str, Any]:
#         query = """
#         SELECT 
#             AVG(execution_time) as avg_time,
#             MIN(execution_time) as min_time,
#             MAX(execution_time) as max_time,
#             COUNT(*) as total_queries,
#             SUM(success) as successful_queries
#         FROM query_metrics
#         WHERE timestamp >= datetime('now', '-24 hours')
#         """
#         with self.metrics_db.connect() as conn:
#             result = conn.execute(query).fetchone()
        
#         return {
#             'avg_execution_time': result[0],
#             'min_execution_time': result[1],
#             'max_execution_time': result[2],
#             'total_queries': result[3],
#             'success_rate': (result[4] / result[3]) * 100 if result[3] > 0 else 0
#         }

# class EnhancedDatabaseAnalyzer(DatabaseAnalyzer):
#     def analyze_data_distribution(self, table_name: str, column_name: str) -> Dict[str, Any]:
#         """Analyze the statistical distribution of data in a column."""
#         query = f"SELECT {column_name} FROM {table_name}"
#         df = pd.read_sql_query(query, self.conn)
        
#         if df[column_name].dtype.kind in 'iuf':  # Numeric data
#             stats = {
#                 'mean': float(df[column_name].mean()),
#                 'median': float(df[column_name].median()),
#                 'std': float(df[column_name].std()),
#                 'min': float(df[column_name].min()),
#                 'max': float(df[column_name].max()),
#                 'histogram': np.histogram(df[column_name], bins=10)
#             }
#         else:  # Categorical data
#             stats = {
#                 'unique_values': len(df[column_name].unique()),
#                 'top_values': df[column_name].value_counts().head().to_dict()
#             }
            
#         return stats

#     def get_table_statistics(self) -> Dict[str, Dict[str, Any]]:
#         """Get comprehensive statistics about all tables."""
#         stats = {}
#         schema = self.get_schema_info()
        
#         for table_name in schema:
#             row_count = self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
#             size_query = f"SELECT page_count * page_size as size FROM pragma_page_count('{table_name}'), pragma_page_size()"
#             size_bytes = self.cursor.execute(size_query).fetchone()[0]
            
#             stats[table_name] = {
#                 'row_count': row_count,
#                 'size_bytes': size_bytes,
#                 'columns': len(schema[table_name]),
#                 'last_analyzed': datetime.now().isoformat()
#             }
            
#         return stats

# class EnhancedLlamaSQLAssistant(LlamaSQLAssistant):
#     def __init__(self, model_path: str, db_analyzer: EnhancedDatabaseAnalyzer):
#         super().__init__(model_path, db_analyzer)
#         self.query_cache = QueryCache()
#         self.validator = QueryValidator()
#         self.optimizer = QueryOptimizer()
#         self.metrics = PerformanceMetrics()
        
#     async def process_query_async(self, user_query: str) -> Dict[str, Any]:
#         try:
#             start_time = time.time()
            
#             # Check cache first
#             cached_result = self.query_cache.get_cached_result(user_query)
#             if cached_result:
#                 return json.loads(cached_result)
            
#             # Generate SQL
#             sql = self.generate_sql(user_query)
            
#             # Validate and optimize
#             is_valid, validation_message = self.validator.validate_syntax(sql)
#             if not is_valid:
#                 raise ValueError(validation_message)
            
#             is_safe, security_message = self.validator.check_security(sql)
#             if not is_safe:
#                 raise ValueError(security_message)
            
#             sql = self.optimizer.optimize_query(sql)
            
#             # Execute query
#             results = self.execute_query(sql)
            
#             execution_time = time.time() - start_time
            
#             # Record metrics
#             self.metrics.record_metric(sql, execution_time, True)
            
#             # Prepare response
#             response = {
#                 'sql': sql,
#                 'results': results.to_dict(orient='records'),
#                 'execution_time': execution_time,
#                 'column_names': list(results.columns),
#                 'row_count': len(results)
#             }
            
#             # Cache the result
#             self.query_cache.cache_result(user_query, json.dumps(response))
            
#             return response
            
#         except Exception as e:
#             error_time = time.time() - start_time
#             self.metrics.record_metric(user_query, error_time, False, str(e))
#             raise

# # FastAPI Models
# class QueryRequest(BaseModel):
#     query: str = Field(..., description="Natural language query to convert to SQL")
#     max_results: int = Field(1000, description="Maximum number of results to return")

# class QueryResponse(BaseModel):
#     sql: str
#     results: List[Dict[str, Any]]
#     execution_time: float
#     column_names: List[str]
#     row_count: int

# def get_assistant():
#     """Factory function to create or return cached assistant instance."""
#     if not hasattr(get_assistant, 'instance'):
#         db_analyzer = EnhancedDatabaseAnalyzer('your_database.db')
#         get_assistant.instance = EnhancedLlamaSQLAssistant('path_to_llama_model.gguf', db_analyzer)
#     return get_assistant.instances

# # FastAPI App
# app = FastAPI(title="Enhanced Llama SQL Assistant API")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API Key authentication
# API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# @app.post("/query", response_model=QueryResponse)
# async def process_query(
#     request: QueryRequest,
#     api_key: str = Depends(API_KEY_HEADER),
#     assistant: EnhancedLlamaSQLAssistant = Depends(get_assistant)
# ):
#     try:
#         return await assistant.process_query_async(request.query)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.get("/metrics")
# async def get_metrics(
#     api_key: str = Depends(API_KEY_HEADER),
#     assistant: EnhancedLlamaSQLAssistant = Depends(get_assistant)
# ):
#     return assistant.metrics.get_performance_report()

# @app.get("/schema")
# async def get_schema(
#     api_key: str = Depends(API_KEY_HEADER),
#     assistant: EnhancedLlamaSQLAssistant = Depends(get_assistant)
# ):
#     return {
#         "schema": assistant.db_analyzer.get_schema_info(),
#         "relationships": assistant.db_analyzer.analyze_relationships(),
#         "statistics": assistant.db_analyzer.get_table_statistics()
#     }

# # WebSocket for real-time query processing
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
    
#     while True:
#         try:
#             query = await websocket.receive_text()
#             assistant = get_assistant()
#             response = await assistant.process_query_async(query)
#             await websocket.send_json(response)
#         except Exception as e:
#             await websocket.send_json({"error": str(e)})

# # Frontend HTML
# @app.get("/")
# async def get_frontend():
#     return """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Llama SQL Assistant</title>
#         <script src="https://cdn.tailwindcss.com"></script>
#         <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
#     </head>
#     <body class="bg-gray-100">
#         <div class="container mx-auto px-4 py-8">
#             <h1 class="text-3xl font-bold mb-8">Llama SQL Assistant</h1>
            
#             <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
#                 <!-- Query Section -->
#                 <div class="bg-white rounded-lg shadow p-6">
#                     <h2 class="text-xl font-semibold mb-4">Query Input</h2>
#                     <textarea id="queryInput" class="w-full h-32 p-2 border rounded" 
#                         placeholder="Enter your query in natural language..."></textarea>
#                     <button onclick="submitQuery()" 
#                         class="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
#                         Submit Query
#                     </button>
#                 </div>

#                 <!-- Results Section -->
#                 <div class="bg-white rounded-lg shadow p-6">
#                     <h2 class="text-xl font-semibold mb-4">Results</h2>
#                     <div id="sqlQuery" class="mb-4 p-2 bg-gray-100 rounded"></div>
#                     <div id="queryResults" class="overflow-x-auto"></div>
#                 </div>
#             </div>

#             <!-- Metrics Section -->
#             <div class="mt-8 bg-white rounded-lg shadow p-6">
#                 <h2 class="text-xl font-semibold mb-4">Performance Metrics</h2>
#                 <div id="metricsChart"></div>
#             </div>
#         </div>

#         <script>
#         const API_KEY = 'your-api-key-here';  // Replace with actual API key

#         async function submitQuery() {
#             const query = document.getElementById('queryInput').value;
            
#             try {
#                 const response = await fetch('/query', {
#                     method: 'POST',
#                     headers: {
#                         'Content-Type': 'application/json',
#                         'X-API-Key': API_KEY
#                     },
#                     body: JSON.stringify({ query })
#                 });
                
#                 const data = await response.json();
                
#                 // Display SQL
#                 document.getElementById('sqlQuery').innerText = data.sql;
                
#                 // Create results table
#                 const table = createResultsTable(data.results, data.column_names);
#                 document.getElementById('queryResults').innerHTML = '';
#                 document.getElementById('queryResults').appendChild(table);
                
#                 // Update metrics
#                 updateMetrics();
                
#             } catch (error) {
#                 console.error('Error:', error);
#                 alert('Error processing query');
#             }
#         }

#         function createResultsTable(results, columns) {
#             const table = document.createElement('table');
#             table.className = 'min-w-full divide-y divide-gray-200';
            
#             // Create header
#             const thead = table.createTHead();
#             const headerRow = thead.insert
#             # Continue the frontend JavaScript
#         function createResultsTable(results, columns) {
#             const table = document.createElement('table');
#             table.className = 'min-w-full divide-y divide-gray-200';
            
#             // Create header
#             const thead = table.createTHead();
#             const headerRow = thead.insertRow();
#             columns.forEach(column => {
#                 const th = document.createElement('th');
#                 th.className = 'px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
#                 th.textContent = column;
#                 headerRow.appendChild(th);
#             });
            
#             // Create body
#             const tbody = table.createTBody();
#             results.forEach(row => {
#                 const tr = tbody.insertRow();
#                 columns.forEach(column => {
#                     const td = tr.insertCell();
#                     td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
#                     td.textContent = row[column];
#                 });
#             });
            
#             return table;
#         }

#         async function updateMetrics() {
#             try {
#                 const response = await fetch('/metrics', {
#                     headers: { 'X-API-Key': API_KEY }
#                 });
#                 const metrics = await response.json();
                
#                 // Create performance chart
#                 const chartData = [{
#                     type: 'indicator',
#                     mode: 'gauge+number',
#                     value: metrics.success_rate,
#                     title: { text: 'Query Success Rate' },
#                     gauge: {
#                         axis: { range: [0, 100] },
#                         bar: { color: 'rgb(59, 130, 246)' },
#                         steps: [
#                             { range: [0, 50], color: 'rgb(239, 68, 68)' },
#                             { range: [50, 80], color: 'rgb(251, 191, 36)' },
#                             { range: [80, 100], color: 'rgb(34, 197, 94)' }
#                         ]
#                     }
#                 }];
                
#                 Plotly.newPlot('metricsChart', chartData);
                
#             } catch (error) {
#                 console.error('Error updating metrics:', error);
#             }
#         }

#         // Initialize metrics on page load
#         updateMetrics();
        
#         // Set up WebSocket connection for real-time updates
#         const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
#         ws.onmessage = (event) => {
#             const data = JSON.parse(event.data);
#             if (data.error) {
#                 alert(`Error: ${data.error}`);
#             } else {
#                 document.getElementById('sqlQuery').innerText = data.sql;
#                 const table = createResultsTable(data.results, data.column_names);
#                 document.getElementById('queryResults').innerHTML = '';
#                 document.getElementById('queryResults').appendChild(table);
#             }
#         };
#     </script>
#     </body>
#     </html>
#     """

# # Additional helper functions


# class QueryHistory:
#     def __init__(self):
#         self.engine = create_engine('sqlite:///query_history.db')
#         self.init_history_table()

#     def init_history_table(self):
#         metadata = MetaData()
#         self.history_table = Table(
#             'query_history', metadata,
#             Column('id', Integer, primary_key=True),
#             Column('query', String),
#             Column('sql', String),
#             Column('timestamp', DateTime),
#             Column('execution_time', Float),
#             Column('success', Integer)
#         )
#         metadata.create_all(self.engine)

#     def add_entry(self, query: str, sql: str, execution_time: float, success: bool):
#         with self.engine.connect() as conn:
#             conn.execute(
#                 self.history_table.insert().values(
#                     query=query,
#                     sql=sql,
#                     timestamp=datetime.now(),
#                     execution_time=execution_time,
#                     success=int(success)
#                 )
#             )

#     def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
#         query = self.history_table.select().order_by(self.history_table.c.timestamp.desc()).limit(limit)
#         with self.engine.connect() as conn:
#             result = conn.execute(query)
#             return [dict(row) for row in result]

# class SchemaVisualizer:
#     def __init__(self, db_analyzer: EnhancedDatabaseAnalyzer):
#         self.db_analyzer = db_analyzer

#     def generate_erd(self) -> str:
#         """Generate Entity Relationship Diagram using Mermaid syntax."""
#         schema = self.db_analyzer.get_schema_info()
#         relationships = self.db_analyzer.analyze_relationships()
        
#         mermaid = ["erDiagram"]
        
#         # Add entities
#         for table, columns in schema.items():
#             entity_def = [f"{table} {{"]
#             for col_name, col_type in columns:
#                 entity_def.append(f"    {col_type} {col_name}")
#             entity_def.append("}")
#             mermaid.append("\n    ".join(entity_def))
        
#         # Add relationships
#         for rel in relationships:
#             mermaid.append(f'{rel["table1"]} ||--o{ rel["table2"]} : has')
            
#         return "\n".join(mermaid)

# # Add new endpoints for the enhanced features
# @app.get("/history")
# async def get_query_history(
#     limit: int = Query(100, ge=1, le=1000),
#     api_key: str = Depends(API_KEY_HEADER)
# ):
#     history = QueryHistory()
#     return history.get_history(limit)

# @app.get("/erd")
# async def get_erd(
#     api_key: str = Depends(API_KEY_HEADER),
#     assistant: EnhancedLlamaSQLAssistant = Depends(get_assistant)
# ):
#     visualizer = SchemaVisualizer(assistant.db_analyzer)
#     return {"erd": visualizer.generate_erd()}

# @app.get("/analyze/{table_name}/{column_name}")
# async def analyze_column(
#     table_name: str,
#     column_name: str,
#     api_key: str = Depends(API_KEY_HEADER),
#     assistant: EnhancedLlamaSQLAssistant = Depends(get_assistant)
# ):
#     return assistant.db_analyzer.analyze_data_distribution(table_name, column_name)

# def run_server():
#     """Run the FastAPI server with uvicorn."""
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     run_server()