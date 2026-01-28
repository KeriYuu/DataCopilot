"""
Insurance domain formulas for training data generation.

Similar to the original type2.py but adapted for insurance metrics.
These formulas are used to:
1. Generate NL2SQL training samples
2. Validate SQL generation correctness
3. Guide the model on proper aggregation methods
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class FormulaCategory(Enum):
    """Categories of insurance formulas."""
    RATIO = "ratio"              # Ratios and percentages
    RATE = "rate"                # Per-unit rates
    AGGREGATION = "aggregation"  # Simple sums/averages


@dataclass
class Formula:
    """Definition of an insurance formula."""
    name: str                    # Formula name (e.g., "loss_ratio")
    display_name: str            # Human-readable name
    category: FormulaCategory
    sql_expression: str          # SQL expression
    description: str             # Plain English description
    numerator_fields: List[str]  # Fields in numerator
    denominator_fields: List[str]  # Fields in denominator
    unit: str                    # Result unit (%, ratio, $, etc.)
    aliases: List[str]           # Alternative names


# Core insurance formulas
FORMULA_DEFINITIONS: List[Formula] = [
    # ==================== RATIO FORMULAS ====================
    Formula(
        name="loss_ratio",
        display_name="Loss Ratio",
        category=FormulaCategory.RATIO,
        sql_expression="SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0)",
        description="Total incurred losses divided by earned premium",
        numerator_fields=["incurred_loss"],
        denominator_fields=["earned_premium"],
        unit="%",
        aliases=["LR", "incurred loss ratio", "loss to premium ratio"]
    ),
    Formula(
        name="paid_loss_ratio",
        display_name="Paid Loss Ratio",
        category=FormulaCategory.RATIO,
        sql_expression="SUM(paid_loss) / NULLIF(SUM(earned_premium), 0)",
        description="Total paid losses divided by earned premium",
        numerator_fields=["paid_loss"],
        denominator_fields=["earned_premium"],
        unit="%",
        aliases=["paid LR", "paid ratio"]
    ),
    Formula(
        name="expense_ratio",
        display_name="Expense Ratio",
        category=FormulaCategory.RATIO,
        sql_expression="SUM(expense) / NULLIF(SUM(written_premium), 0)",
        description="Total expenses divided by written premium",
        numerator_fields=["expense"],
        denominator_fields=["written_premium"],
        unit="%",
        aliases=["ER", "expense to premium"]
    ),
    Formula(
        name="combined_ratio",
        display_name="Combined Ratio",
        category=FormulaCategory.RATIO,
        sql_expression="(SUM(incurred_loss) + SUM(expense)) / NULLIF(SUM(earned_premium), 0)",
        description="Sum of loss ratio and expense ratio",
        numerator_fields=["incurred_loss", "expense"],
        denominator_fields=["earned_premium"],
        unit="%",
        aliases=["CR", "combined loss and expense ratio"]
    ),
    Formula(
        name="reserve_ratio",
        display_name="Reserve Ratio",
        category=FormulaCategory.RATIO,
        sql_expression="SUM(reserved_loss) / NULLIF(SUM(incurred_loss), 0)",
        description="Outstanding reserves as percentage of total incurred",
        numerator_fields=["reserved_loss"],
        denominator_fields=["incurred_loss"],
        unit="%",
        aliases=["reserve to incurred", "outstanding ratio"]
    ),
    
    # ==================== RATE FORMULAS ====================
    Formula(
        name="pure_premium",
        display_name="Pure Premium",
        category=FormulaCategory.RATE,
        sql_expression="SUM(incurred_loss) / NULLIF(SUM(exposure_units), 0)",
        description="Incurred loss per unit of exposure",
        numerator_fields=["incurred_loss"],
        denominator_fields=["exposure_units"],
        unit="$/exposure",
        aliases=["loss cost", "burning cost", "expected loss per exposure"]
    ),
    Formula(
        name="frequency",
        display_name="Claim Frequency",
        category=FormulaCategory.RATE,
        sql_expression="SUM(claim_count) / NULLIF(SUM(exposure_units), 0)",
        description="Number of claims per unit of exposure",
        numerator_fields=["claim_count"],
        denominator_fields=["exposure_units"],
        unit="claims/exposure",
        aliases=["claim rate", "claims per exposure", "frequency rate"]
    ),
    Formula(
        name="severity",
        display_name="Claim Severity",
        category=FormulaCategory.RATE,
        sql_expression="SUM(incurred_loss) / NULLIF(SUM(claim_count), 0)",
        description="Average cost per claim",
        numerator_fields=["incurred_loss"],
        denominator_fields=["claim_count"],
        unit="$/claim",
        aliases=["average claim size", "average severity", "cost per claim"]
    ),
    Formula(
        name="average_premium",
        display_name="Average Premium",
        category=FormulaCategory.RATE,
        sql_expression="SUM(written_premium) / NULLIF(SUM(policy_count), 0)",
        description="Average premium per policy",
        numerator_fields=["written_premium"],
        denominator_fields=["policy_count"],
        unit="$/policy",
        aliases=["premium per policy", "average written premium"]
    ),
    Formula(
        name="rate_per_exposure",
        display_name="Rate per Exposure",
        category=FormulaCategory.RATE,
        sql_expression="SUM(written_premium) / NULLIF(SUM(exposure_units), 0)",
        description="Premium charged per unit of exposure",
        numerator_fields=["written_premium"],
        denominator_fields=["exposure_units"],
        unit="$/exposure",
        aliases=["premium rate", "rate", "charged rate"]
    ),
]


class InsuranceFormulas:
    """
    Insurance formula manager for training data generation.
    
    Similar to type2.py in the original codebase, this class helps:
    1. Parse questions to identify required formulas
    2. Generate correct SQL for formula calculations
    3. Validate formula usage in generated SQL
    """
    
    def __init__(self):
        self._formulas = {f.name: f for f in FORMULA_DEFINITIONS}
        self._alias_map = self._build_alias_map()
    
    def _build_alias_map(self) -> Dict[str, str]:
        """Build mapping from aliases to formula names."""
        alias_map = {}
        for formula in FORMULA_DEFINITIONS:
            alias_map[formula.name.lower()] = formula.name
            alias_map[formula.display_name.lower()] = formula.name
            for alias in formula.aliases:
                alias_map[alias.lower()] = formula.name
        return alias_map
    
    def get_formula(self, name: str) -> Optional[Formula]:
        """Get formula by name or alias."""
        normalized = name.lower().strip()
        formula_name = self._alias_map.get(normalized)
        if formula_name:
            return self._formulas.get(formula_name)
        return None
    
    def detect_formula_in_query(self, query: str) -> List[Formula]:
        """Detect formulas mentioned in a natural language query."""
        query_lower = query.lower()
        detected = []
        
        for alias, formula_name in self._alias_map.items():
            if alias in query_lower:
                formula = self._formulas.get(formula_name)
                if formula and formula not in detected:
                    detected.append(formula)
        
        return detected
    
    def get_sql_for_formula(
        self, 
        formula: Formula,
        group_by: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
    ) -> str:
        """
        Generate SQL query for a formula calculation.
        
        Args:
            formula: Formula to calculate
            group_by: Columns to group by
            where_clause: Optional WHERE conditions
            
        Returns:
            Complete SQL query
        """
        select_parts = []
        
        if group_by:
            select_parts.extend(group_by)
        
        select_parts.append(f"{formula.sql_expression} AS {formula.name}")
        
        sql = f"SELECT {', '.join(select_parts)}\nFROM insurance_loss_data"
        
        if where_clause:
            sql += f"\nWHERE {where_clause}"
        
        if group_by:
            sql += f"\nGROUP BY {', '.join(group_by)}"
            sql += f"\nORDER BY {formula.name} DESC"
        
        sql += "\nLIMIT 1000"
        
        return sql
    
    def validate_sql_formula(self, sql: str, expected_formula: Formula) -> Tuple[bool, str]:
        """
        Validate that SQL correctly implements a formula.
        
        Args:
            sql: SQL query to validate
            expected_formula: Expected formula
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_upper = sql.upper()
        
        # Check for proper aggregation (not AVG of ratios)
        if "AVG(" in sql_upper and expected_formula.category == FormulaCategory.RATIO:
            return False, "Should use SUM/SUM for ratios, not AVG"
        
        # Check required fields are present
        for field in expected_formula.numerator_fields:
            if field.upper() not in sql_upper:
                return False, f"Missing numerator field: {field}"
        
        for field in expected_formula.denominator_fields:
            if field.upper() not in sql_upper:
                return False, f"Missing denominator field: {field}"
        
        # Check for NULLIF to prevent division by zero
        if "/" in sql and "NULLIF" not in sql_upper:
            return False, "Missing NULLIF for division by zero protection"
        
        return True, ""
    
    def get_all_formulas(self) -> List[Formula]:
        """Get all defined formulas."""
        return FORMULA_DEFINITIONS
    
    def get_formulas_by_category(self, category: FormulaCategory) -> List[Formula]:
        """Get formulas by category."""
        return [f for f in FORMULA_DEFINITIONS if f.category == category]


# Helper functions for training data generation
def is_formula_question(query: str) -> bool:
    """Check if a query requires formula calculation."""
    formulas = InsuranceFormulas()
    detected = formulas.detect_formula_in_query(query)
    return len(detected) > 0


def is_growth_rate_question(query: str) -> bool:
    """Check if a query asks for growth/change calculation."""
    growth_keywords = [
        "growth", "change", "increase", "decrease", "yoy", "year over year",
        "compared to", "versus", "vs", "difference", "trend"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in growth_keywords)


def get_formula_sql_hint(query: str) -> Optional[str]:
    """Get SQL hint for formula questions."""
    formulas = InsuranceFormulas()
    detected = formulas.detect_formula_in_query(query)
    
    if detected:
        hints = []
        for formula in detected:
            hints.append(f"-- {formula.display_name}: {formula.sql_expression}")
        return "\n".join(hints)
    
    return None
