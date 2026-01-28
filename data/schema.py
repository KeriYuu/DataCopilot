"""
Insurance data schema definition.

This module defines the structure of the insurance loss data table
that the Data Copilot queries.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


TABLE_NAME = "insurance_loss_data"


class ColumnType(Enum):
    """Column data types for schema definition."""
    STRING = "String"
    INT = "Int64"
    FLOAT = "Float64"
    DATE = "Date"


@dataclass
class ColumnInfo:
    """Information about a single column."""
    name: str
    dtype: ColumnType
    description: str
    sample_values: Optional[List[str]] = None
    is_metric: bool = False
    is_dimension: bool = False


class InsuranceSchema:
    """
    Insurance loss data schema definition.
    
    The schema represents a denormalized "Wide Table" containing 
    approximately 5 million records of historical insurance loss information.
    """
    
    # Identifier columns
    IDENTIFIERS: List[ColumnInfo] = [
        ColumnInfo(
            name="policy_id",
            dtype=ColumnType.STRING,
            description="Unique identifier for the insurance policy",
            sample_values=["POL-2023-001234", "POL-2022-567890"],
            is_dimension=True
        ),
        ColumnInfo(
            name="claim_id",
            dtype=ColumnType.STRING,
            description="Unique identifier for the claim",
            sample_values=["CLM-2023-001", "CLM-2022-999"],
            is_dimension=True
        ),
    ]
    
    # Dimension columns (for grouping/filtering)
    DIMENSIONS: List[ColumnInfo] = [
        ColumnInfo(
            name="state",
            dtype=ColumnType.STRING,
            description="US State code (2-letter abbreviation)",
            sample_values=["CA", "NY", "TX", "FL", "IL"],
            is_dimension=True
        ),
        ColumnInfo(
            name="zip_code",
            dtype=ColumnType.STRING,
            description="5-digit ZIP code",
            sample_values=["90001", "10001", "60601"],
            is_dimension=True
        ),
        ColumnInfo(
            name="class_code",
            dtype=ColumnType.STRING,
            description="Standard Industry Classification code (NCCI class code)",
            sample_values=["8810", "5506", "8742", "8820"],
            is_dimension=True
        ),
        ColumnInfo(
            name="class_group",
            dtype=ColumnType.STRING,
            description="High-level classification group",
            sample_values=["WC", "GL", "AUTO", "PROPERTY"],
            is_dimension=True
        ),
        ColumnInfo(
            name="coverage_type",
            dtype=ColumnType.STRING,
            description="Type of insurance coverage",
            sample_values=["Workers Compensation", "General Liability", "Commercial Auto"],
            is_dimension=True
        ),
        ColumnInfo(
            name="policy_year",
            dtype=ColumnType.INT,
            description="Policy effective year",
            sample_values=["2020", "2021", "2022", "2023"],
            is_dimension=True
        ),
        ColumnInfo(
            name="accident_year",
            dtype=ColumnType.INT,
            description="Year when the accident/loss occurred",
            sample_values=["2020", "2021", "2022", "2023"],
            is_dimension=True
        ),
        ColumnInfo(
            name="industry",
            dtype=ColumnType.STRING,
            description="Industry classification",
            sample_values=["Construction", "Manufacturing", "Healthcare", "Retail"],
            is_dimension=True
        ),
        ColumnInfo(
            name="region",
            dtype=ColumnType.STRING,
            description="Geographic region",
            sample_values=["West", "Northeast", "Midwest", "South"],
            is_dimension=True
        ),
    ]
    
    # Metric columns (for aggregation)
    METRICS: List[ColumnInfo] = [
        ColumnInfo(
            name="written_premium",
            dtype=ColumnType.FLOAT,
            description="Total written premium amount in USD",
            is_metric=True
        ),
        ColumnInfo(
            name="earned_premium",
            dtype=ColumnType.FLOAT,
            description="Earned premium amount in USD",
            is_metric=True
        ),
        ColumnInfo(
            name="incurred_loss",
            dtype=ColumnType.FLOAT,
            description="Total incurred loss amount in USD (paid + reserved)",
            is_metric=True
        ),
        ColumnInfo(
            name="paid_loss",
            dtype=ColumnType.FLOAT,
            description="Paid loss amount in USD",
            is_metric=True
        ),
        ColumnInfo(
            name="reserved_loss",
            dtype=ColumnType.FLOAT,
            description="Reserved (outstanding) loss amount in USD",
            is_metric=True
        ),
        ColumnInfo(
            name="exposure_units",
            dtype=ColumnType.FLOAT,
            description="Number of exposure units (e.g., payroll in $100s for WC)",
            is_metric=True
        ),
        ColumnInfo(
            name="claim_count",
            dtype=ColumnType.INT,
            description="Number of claims",
            is_metric=True
        ),
        ColumnInfo(
            name="policy_count",
            dtype=ColumnType.INT,
            description="Number of policies",
            is_metric=True
        ),
    ]
    
    @classmethod
    def all_columns(cls) -> List[ColumnInfo]:
        """Get all columns."""
        return cls.IDENTIFIERS + cls.DIMENSIONS + cls.METRICS
    
    @classmethod
    def column_names(cls) -> List[str]:
        """Get all column names."""
        return [col.name for col in cls.all_columns()]
    
    @classmethod
    def dimension_names(cls) -> List[str]:
        """Get dimension column names."""
        return [col.name for col in cls.DIMENSIONS + cls.IDENTIFIERS]
    
    @classmethod
    def metric_names(cls) -> List[str]:
        """Get metric column names."""
        return [col.name for col in cls.METRICS]
    
    @classmethod
    def get_column_info(cls, name: str) -> Optional[ColumnInfo]:
        """Get column info by name."""
        for col in cls.all_columns():
            if col.name == name:
                return col
        return None


# Schema description for prompts
SCHEMA_DESCRIPTION = f"""
Table Name: {TABLE_NAME}

== IDENTIFIERS ==
{chr(10).join([f"- {c.name} ({c.dtype.value}): {c.description}" for c in InsuranceSchema.IDENTIFIERS])}

== DIMENSIONS (for filtering/grouping) ==
{chr(10).join([f"- {c.name} ({c.dtype.value}): {c.description}. Examples: {c.sample_values}" for c in InsuranceSchema.DIMENSIONS])}

== METRICS (for aggregation) ==
{chr(10).join([f"- {c.name} ({c.dtype.value}): {c.description}" for c in InsuranceSchema.METRICS])}

== COMMON FORMULAS ==
- Loss Ratio = SUM(incurred_loss) / SUM(earned_premium) -- NOT average of ratios!
- Pure Premium = SUM(incurred_loss) / SUM(exposure_units)
- Frequency = SUM(claim_count) / SUM(exposure_units)
- Severity = SUM(incurred_loss) / SUM(claim_count)
- Average Premium = SUM(written_premium) / SUM(policy_count)
"""


# Entity mappings for keyword rewriting
ENTITY_MAPPINGS: Dict[str, Dict[str, str]] = {
    "coverage_type": {
        "workers comp": "Workers Compensation",
        "wc": "Workers Compensation",
        "work comp": "Workers Compensation",
        "gl": "General Liability",
        "general liability": "General Liability",
        "auto": "Commercial Auto",
        "commercial auto": "Commercial Auto",
        "property": "Commercial Property",
    },
    "class_group": {
        "workers comp": "WC",
        "wc": "WC",
        "general liability": "GL",
        "gl": "GL",
        "auto": "AUTO",
        "property": "PROPERTY",
    },
    "region": {
        "socal": "West",
        "southern california": "West",
        "norcal": "West",
        "northern california": "West",
        "california": "West",
        "west coast": "West",
        "east coast": "Northeast",
        "new england": "Northeast",
        "midwest": "Midwest",
        "south": "South",
        "southeast": "South",
    },
    "state": {
        "california": "CA",
        "new york": "NY",
        "texas": "TX",
        "florida": "FL",
        "illinois": "IL",
        "pennsylvania": "PA",
        "ohio": "OH",
        "georgia": "GA",
        "north carolina": "NC",
        "michigan": "MI",
    }
}


# Common class codes reference
CLASS_CODE_REFERENCE: Dict[str, str] = {
    "8810": "Clerical Office Employees",
    "8742": "Salespersons - Outside",
    "5506": "Street or Road Construction",
    "8820": "Attorneys",
    "8017": "Store - Retail",
    "9015": "Building Operation - Commercial",
    "5183": "Plumbing",
    "5190": "Electrical Wiring",
}
