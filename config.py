import os
from pathlib import Path

class Settings:
    PROJECT_NAME = "SQL Assistant"
    VERSION = "0.1.0"
    API_V1_STR = "/api/v1"

    # Database configuration
    DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # Default to SQLite
    if DB_TYPE == "sqlite":
        DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")  # SQLite connection string
    elif DB_TYPE == "postgresql":
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/dbname")  # PostgreSQL connection string
    elif DB_TYPE == "mysql":
        DATABASE_URL = os.getenv("DATABASE_URL", "mysql://username:password@localhost/dbname")  # MySQL connection string
    else:
        raise ValueError(f"Unsupported database type: {DB_TYPE}")

    MODEL_PATH = os.getenv("MODEL_PATH", "path_to_llama_model.gguf")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Security
    API_KEY = os.getenv("API_KEY", "your-secret-api-key")  # Default key for development

settings = Settings()

# Example: Printing the selected database URL for verification
print(f"Using database: {settings.DB_TYPE}")
print(f"Database URL: {settings.DATABASE_URL}")
