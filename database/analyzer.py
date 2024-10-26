

import sqlite3
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import sqlparse
import logging

class DatabaseAnalyzer:
    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def get_schema_info(self) -> Dict[str, List[Tuple[str, str]]]:
        """Extract database schema including tables, columns, and their types."""
        schema = {}
        
        # Get all tables
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()
            schema[table_name] = [(col[1], col[2]) for col in columns]  # (column_name, type)
            
        return schema
    
    def analyze_relationships(self) -> List[Dict]:
        """Identify potential foreign key relationships based on column names."""
        relationships = []
        schema = self.get_schema_info()
        
        for table1 in schema:
            for table2 in schema:
                if table1 != table2:
                    for col1_name, _ in schema[table1]:
                        for col2_name, _ in schema[table2]:
                            # Check for potential relationships (e.g., id columns)
                            if (f"{table2.lower()}_id" == col1_name.lower() or
                                f"{table1.lower()}_id" == col2_name.lower()):
                                relationships.append({
                                    'table1': table1,
                                    'column1': col1_name,
                                    'table2': table2,
                                    'column2': col2_name
                                })
        return relationships

class EnhancedDatabaseAnalyzer(DatabaseAnalyzer):
    def analyze_data_distribution(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Analyze the statistical distribution of data in a column."""
        query = f"SELECT {column_name} FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        
        if df[column_name].dtype.kind in 'iuf':  # Numeric data
            stats = {
                'mean': float(df[column_name].mean()),
                'median': float(df[column_name].median()),
                'std': float(df[column_name].std()),
                'min': float(df[column_name].min()),
                'max': float(df[column_name].max())
            }
        else:  # Categorical data
            stats = {
                'unique_values': len(df[column_name].unique()),
                'top_values': df[column_name].value_counts().head().to_dict()
            }
            
        return stats

    def get_table_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive statistics about all tables."""
        stats = {}
        schema = self.get_schema_info()
        
        for table_name in schema:
            row_count = self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            stats[table_name] = {
                'row_count': row_count,
                'columns': len(schema[table_name]),
                'last_analyzed': datetime.now().isoformat()
            }
            
        return stats
