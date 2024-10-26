import sqlparse
from typing import Tuple

class QueryValidator:
    @staticmethod
    def validate_syntax(sql: str) -> Tuple[bool, str]:
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, "Empty query"
            return True, "Valid SQL syntax"
        except Exception as e:
            return False, f"Syntax error: {str(e)}"
    
    @staticmethod
    def check_security(sql: str) -> Tuple[bool, str]:
        sql_lower = sql.lower()
        dangerous_keywords = ['drop', 'truncate', 'delete', 'update', 'insert', 'alter', 'create']
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False, f"Unauthorized {keyword.upper()} operation detected"
        return True, "Query passed security check"