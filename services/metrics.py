from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime
from datetime import datetime

class PerformanceMetrics:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.init_metrics_table()
    
    def init_metrics_table(self):
        metadata = MetaData()
        self.metrics_table = Table(
            'query_metrics', metadata,
            Column('id', Integer, primary_key=True),
            Column('query_hash', String),
            Column('execution_time', Float),
            Column('timestamp', DateTime),
            Column('success', Integer),
            Column('error_message', String, nullable=True)
        )
        metadata.create_all(self.engine)
