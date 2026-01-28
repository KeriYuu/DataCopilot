"""
SQL utility functions for validation and formatting.
"""
import re
from typing import List, Optional, Tuple
from loguru import logger

from data_copilot.data.schema import InsuranceSchema, TABLE_NAME


class SQLValidator:
    """Validates and sanitizes generated SQL queries."""
    
    # Dangerous SQL patterns to block
    DANGEROUS_PATTERNS = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bTRUNCATE\b',
        r'\bINSERT\b',
        r'\bUPDATE\b',
        r'\bALTER\b',
        r'\bCREATE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'--',  # SQL comments (potential injection)
        r'/\*',  # Block comments
        r'\bEXEC\b',
        r'\bEXECUTE\b',
    ]
    
    def __init__(self):
        self.valid_columns = set(InsuranceSchema.column_names())
        self.table_name = TABLE_NAME
    
    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_upper = sql.upper()
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return False, f"Potentially dangerous SQL pattern detected: {pattern}"
        
        # Must be a SELECT query
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Check table name
        if self.table_name.lower() not in sql.lower():
            return False, f"Query must reference table: {self.table_name}"
        
        # Extract and validate column references
        invalid_cols = self._find_invalid_columns(sql)
        if invalid_cols:
            return False, f"Invalid column references: {', '.join(invalid_cols)}"
        
        return True, None
    
    def _find_invalid_columns(self, sql: str) -> List[str]:
        """Find any column references that don't exist in schema."""
        # This is a simplified check - a proper parser would be better
        # Extract potential column names (words not in SQL keywords)
        sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE',
            'BETWEEN', 'IS', 'NULL', 'AS', 'GROUP', 'BY', 'ORDER', 'ASC',
            'DESC', 'LIMIT', 'OFFSET', 'HAVING', 'JOIN', 'ON', 'LEFT',
            'RIGHT', 'INNER', 'OUTER', 'FULL', 'CROSS', 'UNION', 'ALL',
            'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'NULLIF',
            'COALESCE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'CAST',
            'TRUE', 'FALSE', self.table_name.upper()
        }
        
        # Simple word extraction - not perfect but catches common issues
        words = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql)
        
        invalid = []
        for word in words:
            word_upper = word.upper()
            if word_upper not in sql_keywords:
                if word.lower() not in [c.lower() for c in self.valid_columns]:
                    # Could be an alias, skip common patterns
                    if not word.startswith(('t', 'a', 'b', 'c', 'x', 'y')):
                        invalid.append(word)
        
        return list(set(invalid))
    
    def sanitize(self, sql: str) -> str:
        """
        Sanitize SQL by adding safety measures.
        
        Args:
            sql: SQL query string
            
        Returns:
            Sanitized SQL
        """
        sql = sql.strip()
        
        # Remove any trailing semicolons (ClickHouse doesn't need them)
        sql = sql.rstrip(';')
        
        # Add LIMIT if not present
        if 'LIMIT' not in sql.upper():
            sql = f"{sql}\nLIMIT 1000"
        
        return sql


def format_sql_result(results: List[dict], max_display: int = 50) -> str:
    """
    Format SQL results for display.
    
    Args:
        results: List of result dictionaries
        max_display: Maximum rows to display
        
    Returns:
        Formatted string representation
    """
    if not results:
        return "No results returned."
    
    # Get column widths
    headers = list(results[0].keys())
    col_widths = {h: len(h) for h in headers}
    
    display_results = results[:max_display]
    
    for row in display_results:
        for h in headers:
            val_len = len(str(row.get(h, '')))
            col_widths[h] = min(max(col_widths[h], val_len), 50)  # Cap at 50 chars
    
    # Build table
    lines = []
    
    # Header
    header_line = " | ".join(h.ljust(col_widths[h])[:col_widths[h]] for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Rows
    for row in display_results:
        row_line = " | ".join(
            str(row.get(h, '')).ljust(col_widths[h])[:col_widths[h]] 
            for h in headers
        )
        lines.append(row_line)
    
    # Footer
    if len(results) > max_display:
        lines.append(f"\n... {len(results) - max_display} more rows not shown")
    
    lines.append(f"\nTotal: {len(results)} rows")
    
    return "\n".join(lines)


def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL from a model response that might contain markdown.
    
    Args:
        response: Model response potentially containing SQL in code blocks
        
    Returns:
        Extracted SQL query
    """
    # Try to extract from markdown code blocks
    sql_match = re.search(r'```(?:sql)?\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to find SELECT statement
    select_match = re.search(r'(SELECT\s+.*)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        # Clean up any trailing text
        sql = re.split(r'\n\n|\n[A-Z][a-z]', sql)[0]
        return sql.strip()
    
    return response.strip()
