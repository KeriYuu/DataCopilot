"""
SQL Executor tool for ClickHouse database operations.

Executes validated SQL queries against the insurance data warehouse.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from config import settings


@dataclass
class ExecutionResult:
    """Result of SQL execution."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    row_count: int = 0
    execution_time_ms: float = 0.0
    sql: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "sql": self.sql,
        }


class SQLExecutor:
    """
    Executes SQL queries against ClickHouse database.
    
    Supports both ClickHouse native driver and HTTP interface.
    Falls back to pandas DataFrame for local development.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        use_dataframe: bool = False,
        dataframe: Any = None,
    ):
        """
        Initialize the SQL executor.
        
        Args:
            host: ClickHouse host
            port: ClickHouse port
            database: Database name
            user: Username
            password: Password
            use_dataframe: If True, execute queries against a pandas DataFrame
            dataframe: Pandas DataFrame to use (if use_dataframe is True)
        """
        self.host = host or settings.clickhouse_host
        self.port = port or settings.clickhouse_port
        self.database = database or settings.clickhouse_database
        self.user = user or settings.clickhouse_user
        self.password = password or settings.clickhouse_password
        
        self.use_dataframe = use_dataframe
        self._dataframe = dataframe
        self._client = None
        
    def _get_client(self):
        """Get or create ClickHouse client."""
        if self._client is None and not self.use_dataframe:
            try:
                import clickhouse_connect
                self._client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    username=self.user,
                    password=self.password,
                )
            except ImportError:
                logger.warning("clickhouse-connect not installed, using DataFrame mode")
                self.use_dataframe = True
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse: {e}")
                raise
        return self._client
    
    def execute(self, sql: str) -> ExecutionResult:
        """
        Execute a SQL query.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            ExecutionResult with data or error
        """
        start_time = time.time()
        
        try:
            if self.use_dataframe:
                result = self._execute_dataframe(sql)
            else:
                result = self._execute_clickhouse(sql)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                data=result,
                row_count=len(result) if result else 0,
                execution_time_ms=execution_time,
                sql=sql,
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"SQL execution failed: {e}")
            
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                sql=sql,
            )
    
    def _execute_clickhouse(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL against ClickHouse."""
        client = self._get_client()
        result = client.query(sql)
        
        # Convert to list of dicts
        columns = result.column_names
        data = []
        for row in result.result_rows:
            data.append(dict(zip(columns, row)))
        
        return data
    
    def _execute_dataframe(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL against a pandas DataFrame using pandasql.
        
        This is a fallback for local development and testing.
        """
        if self._dataframe is None:
            raise ValueError("No DataFrame provided for execution")
        
        try:
            import pandasql as ps
        except ImportError:
            raise ImportError("pandasql required for DataFrame mode. Install with: pip install pandasql")
        
        # pandasql expects table name to match DataFrame variable name
        # We need to inject our dataframe into the local environment
        from data.schema import TABLE_NAME
        
        # Replace table name with 'df' for pandasql
        modified_sql = sql.replace(TABLE_NAME, 'df')
        
        # Execute query
        df = self._dataframe
        result_df = ps.sqldf(modified_sql, {'df': df})
        
        return result_df.to_dict('records')
    
    def set_dataframe(self, df) -> None:
        """Set the DataFrame for DataFrame mode."""
        self._dataframe = df
        self.use_dataframe = True
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test database connection.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.use_dataframe:
                if self._dataframe is not None:
                    return True, f"DataFrame mode active with {len(self._dataframe)} rows"
                else:
                    return False, "No DataFrame loaded"
            else:
                client = self._get_client()
                result = client.query("SELECT 1")
                return True, f"Connected to ClickHouse at {self.host}:{self.port}"
        except Exception as e:
            return False, str(e)


# Global executor instance
_executor: Optional[SQLExecutor] = None


def get_executor() -> SQLExecutor:
    """Get or create the global SQL executor."""
    global _executor
    if _executor is None:
        _executor = SQLExecutor()
    return _executor


def set_executor(executor: SQLExecutor) -> None:
    """Set the global SQL executor."""
    global _executor
    _executor = executor
