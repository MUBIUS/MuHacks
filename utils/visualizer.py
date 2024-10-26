class SchemaVisualizer:
    def __init__(self, db_analyzer):
        self.db_analyzer = db_analyzer
    
    def generate_erd(self) -> str:
        schema = self.db_analyzer.get_schema_info()
        relationships = self.db_analyzer.analyze_relationships()
        
        mermaid = ["erDiagram"]
        for table, columns in schema.items():
            entity_def = [f"{table} {{"]
            for col_name, col_type in columns:
                entity_def.append(f"    {col_type} {col_name}")
            entity_def.append("}")
            mermaid.append("\n    ".join(entity_def))
            
        return "\n".join(mermaid)