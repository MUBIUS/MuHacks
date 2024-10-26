import redis
import json
import hashlib
from typing import Optional

class QueryCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = 3600
        
    def get_cache_key(self, query: str) -> str:
        return f"sql_query:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get_cached_result(self, query: str) -> Optional[str]:
        return self.redis_client.get(self.get_cache_key(query))
    
    def cache_result(self, query: str, result: str):
        self.redis_client.setex(self.get_cache_key(query), self.ttl, result)