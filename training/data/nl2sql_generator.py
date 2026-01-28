"""
NL2SQL training data generator.

Generates training samples for converting natural language queries
to ClickHouse-compatible SQL, with domain-specific insurance formulas.
"""
import json
import random
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from tools.sql_executor import SQLExecutor

from training.domain.formulas import (
    InsuranceFormulas, FormulaCategory, Formula, FORMULA_DEFINITIONS
)
from training.domain.terminology import (
    InsuranceTerminology, STATE_MAPPINGS, CLASS_CODE_TERMS
)


TABLE_NAME = "insurance_loss_data"


@dataclass
class NL2SQLSample:
    """A single NL2SQL training sample."""
    query: str
    sql: str
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "instruction": f"""Convert the following insurance data query to ClickHouse SQL.
Table: {TABLE_NAME}
Columns: policy_id, claim_id, state, zip_code, class_code, class_group, coverage_type, 
         policy_year, accident_year, industry, region, written_premium, earned_premium,
         incurred_loss, paid_loss, reserved_loss, exposure_units, claim_count, policy_count

Important: 
- Use SUM/SUM for ratios (NOT AVG of ratios)
- Use NULLIF for division to prevent divide by zero
- Include appropriate GROUP BY for aggregations""",
            "input": self.query,
            "output": self.sql
        }


# SQL templates for different query types
SQL_TEMPLATES = {
    "simple_aggregation": """SELECT {select_clause}
FROM {table}
{where_clause}
{group_by}
{order_by}
LIMIT {limit}""",
    
    "formula_calculation": """SELECT {dimensions},
    {formula_sql} AS {formula_name}
FROM {table}
{where_clause}
GROUP BY {dimensions}
ORDER BY {formula_name} DESC
LIMIT {limit}""",
    
    "top_n": """SELECT {dimension}, {metric}
FROM {table}
{where_clause}
ORDER BY {metric} DESC
LIMIT {n}""",
    
    "comparison": """SELECT {dimension},
    SUM({metric}) AS total_{metric}
FROM {table}
{where_clause}
GROUP BY {dimension}
ORDER BY total_{metric} DESC""",
    
    "growth": """SELECT 
    policy_year,
    SUM({metric}) AS {metric},
    (SUM({metric}) - LAG(SUM({metric})) OVER (ORDER BY policy_year)) / 
        NULLIF(LAG(SUM({metric})) OVER (ORDER BY policy_year), 0) AS growth_rate
FROM {table}
{where_clause}
GROUP BY policy_year
ORDER BY policy_year""",
}


# Natural language query patterns with corresponding SQL patterns
QUERY_PATTERNS = [
    # Simple aggregations
    {
        "nl": "What is the total {metric} for {state}?",
        "sql": "SELECT SUM({metric}) AS total_{metric}\nFROM {table}\nWHERE state = '{state_code}'",
    },
    {
        "nl": "Show total {metric} by state",
        "sql": "SELECT state, SUM({metric}) AS total_{metric}\nFROM {table}\nGROUP BY state\nORDER BY total_{metric} DESC\nLIMIT 1000",
    },
    {
        "nl": "Calculate average {metric} for class code {class_code}",
        "sql": "SELECT AVG({metric}) AS avg_{metric}\nFROM {table}\nWHERE class_code = '{class_code}'",
    },
    
    # Formula calculations
    {
        "nl": "What is the loss ratio for {state}?",
        "sql": "SELECT SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0) AS loss_ratio\nFROM {table}\nWHERE state = '{state_code}'",
    },
    {
        "nl": "Calculate the loss ratio by state",
        "sql": "SELECT state,\n    SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0) AS loss_ratio\nFROM {table}\nGROUP BY state\nORDER BY loss_ratio DESC\nLIMIT 1000",
    },
    {
        "nl": "What is the pure premium for class code {class_code} in {state}?",
        "sql": "SELECT SUM(incurred_loss) / NULLIF(SUM(exposure_units), 0) AS pure_premium\nFROM {table}\nWHERE class_code = '{class_code}' AND state = '{state_code}'",
    },
    {
        "nl": "Show claim frequency by coverage type",
        "sql": "SELECT coverage_type,\n    SUM(claim_count) / NULLIF(SUM(exposure_units), 0) AS frequency\nFROM {table}\nGROUP BY coverage_type\nORDER BY frequency DESC\nLIMIT 1000",
    },
    {
        "nl": "Calculate severity for {coverage} in {year}",
        "sql": "SELECT SUM(incurred_loss) / NULLIF(SUM(claim_count), 0) AS severity\nFROM {table}\nWHERE coverage_type = '{coverage}' AND policy_year = {year}",
    },
    {
        "nl": "What is the combined ratio by region?",
        "sql": "SELECT region,\n    (SUM(incurred_loss) + SUM(expense)) / NULLIF(SUM(earned_premium), 0) AS combined_ratio\nFROM {table}\nGROUP BY region\nORDER BY combined_ratio DESC\nLIMIT 1000",
    },
    
    # Top N queries
    {
        "nl": "Show top 10 states by {metric}",
        "sql": "SELECT state, SUM({metric}) AS total_{metric}\nFROM {table}\nGROUP BY state\nORDER BY total_{metric} DESC\nLIMIT 10",
    },
    {
        "nl": "Which class codes have the highest loss ratio?",
        "sql": "SELECT class_code,\n    SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0) AS loss_ratio\nFROM {table}\nGROUP BY class_code\nHAVING SUM(earned_premium) > 0\nORDER BY loss_ratio DESC\nLIMIT 20",
    },
    {
        "nl": "Top 5 industries by claim count in {state}",
        "sql": "SELECT industry, SUM(claim_count) AS total_claims\nFROM {table}\nWHERE state = '{state_code}'\nGROUP BY industry\nORDER BY total_claims DESC\nLIMIT 5",
    },
    
    # Filtered queries
    {
        "nl": "Show {metric} for Workers Compensation in {state}",
        "sql": "SELECT SUM({metric}) AS total_{metric}\nFROM {table}\nWHERE coverage_type = 'Workers Compensation' AND state = '{state_code}'",
    },
    {
        "nl": "Calculate loss ratio for {year} by state",
        "sql": "SELECT state,\n    SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0) AS loss_ratio\nFROM {table}\nWHERE policy_year = {year}\nGROUP BY state\nORDER BY loss_ratio DESC\nLIMIT 1000",
    },
    {
        "nl": "What is the average premium for class {class_code} in the {region} region?",
        "sql": "SELECT SUM(written_premium) / NULLIF(SUM(policy_count), 0) AS avg_premium\nFROM {table}\nWHERE class_code = '{class_code}' AND region = '{region}'",
    },
    
    # Count queries
    {
        "nl": "How many claims in {state} for {year}?",
        "sql": "SELECT SUM(claim_count) AS total_claims\nFROM {table}\nWHERE state = '{state_code}' AND policy_year = {year}",
    },
    {
        "nl": "Count policies by coverage type",
        "sql": "SELECT coverage_type, SUM(policy_count) AS policy_count\nFROM {table}\nGROUP BY coverage_type\nORDER BY policy_count DESC\nLIMIT 1000",
    },
    
    # Trend queries
    {
        "nl": "Show {metric} trend by year for {state}",
        "sql": "SELECT policy_year, SUM({metric}) AS {metric}\nFROM {table}\nWHERE state = '{state_code}'\nGROUP BY policy_year\nORDER BY policy_year",
    },
    {
        "nl": "Compare loss ratio across years",
        "sql": "SELECT policy_year,\n    SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0) AS loss_ratio\nFROM {table}\nGROUP BY policy_year\nORDER BY policy_year",
    },
]


class NL2SQLDataGenerator:
    """
    Generator for NL2SQL training data.
    
    Creates training samples with correct insurance formula implementations.
    """
    
    def __init__(self):
        self.formulas = InsuranceFormulas()
        self.terminology = InsuranceTerminology()
        
        # Sample values
        self.states = list(STATE_MAPPINGS.values())[:20]  # Top 20 states
        self.state_names = {v: k for k, v in STATE_MAPPINGS.items()}
        self.years = [2019, 2020, 2021, 2022, 2023]
        self.class_codes = [t.name for t in CLASS_CODE_TERMS]
        self.coverages = ["Workers Compensation", "General Liability", "Commercial Auto", "Commercial Property"]
        self.regions = ["West", "Northeast", "Midwest", "South"]
        self.metrics = ["written_premium", "earned_premium", "incurred_loss", "paid_loss", 
                       "claim_count", "exposure_units", "policy_count"]
        self.industries = ["Construction", "Manufacturing", "Healthcare", "Retail", "Technology", "Hospitality"]
        self.class_groups = ["WC", "GL", "AUTO", "PROPERTY"]

    def generate_dataframe(self, n_rows: int = 5000, seed: int = 42):
        """
        Generate a synthetic DataFrame for NL2SQL ground truth evaluation.
        
        Args:
            n_rows: Number of rows to generate
            seed: Random seed for reproducibility
            
        Returns:
            Pandas DataFrame with insurance_loss_data schema
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for DataFrame generation. Install with: pip install pandas") from e
        
        rng = random.Random(seed)
        
        rows: List[Dict[str, Any]] = []
        for _ in range(n_rows):
            policy_year = rng.choice(self.years)
            accident_year = rng.choice(self.years)
            state = rng.choice(self.states)
            class_code = rng.choice(self.class_codes)
            coverage = rng.choice(self.coverages)
            region = rng.choice(self.regions)
            class_group = rng.choice(self.class_groups)
            
            rows.append({
                "policy_id": f"POL-{policy_year}-{rng.randint(100000, 999999)}",
                "claim_id": f"CLM-{accident_year}-{rng.randint(1000, 9999)}",
                "state": state,
                "zip_code": f"{rng.randint(10000, 99999)}",
                "class_code": class_code,
                "class_group": class_group,
                "coverage_type": coverage,
                "policy_year": int(policy_year),
                "accident_year": int(accident_year),
                "industry": rng.choice(self.industries),
                "region": region,
                "written_premium": round(rng.uniform(1000, 500000), 2),
                "earned_premium": round(rng.uniform(1000, 500000), 2),
                "incurred_loss": round(rng.uniform(0, 400000), 2),
                "paid_loss": round(rng.uniform(0, 300000), 2),
                "reserved_loss": round(rng.uniform(0, 200000), 2),
                "exposure_units": round(rng.uniform(1, 10000), 2),
                "claim_count": rng.randint(1, 50),
                "policy_count": rng.randint(1, 100),
                "expense": round(rng.uniform(0, 100000), 2),
            })
        
        return pd.DataFrame(rows)

    def _serialize_result_data(self, data: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert execution result data to JSON-serializable types."""
        if data is None:
            return None
        
        try:
            import numpy as np
        except ImportError:
            np = None
        
        serialized = []
        for row in data:
            clean_row = {}
            for key, value in row.items():
                if np is not None:
                    if isinstance(value, np.integer):
                        value = int(value)
                    elif isinstance(value, np.floating):
                        value = float(value)
                clean_row[key] = value
            serialized.append(clean_row)
        
        return serialized
    
    def _get_state_name(self, code: str) -> str:
        """Get full state name for a code."""
        for name, c in STATE_MAPPINGS.items():
            if c == code:
                return name.title()
        return code
    
    def generate_from_patterns(self, n_per_pattern: int = 50) -> List[NL2SQLSample]:
        """Generate samples from predefined patterns."""
        samples = []
        
        for pattern in QUERY_PATTERNS:
            for _ in range(n_per_pattern):
                # Pre-select all random values ONCE to ensure consistency between NL and SQL
                state_code = random.choice(self.states)
                state_name = self._get_state_name(state_code)
                metric = random.choice(self.metrics)
                class_code = random.choice(self.class_codes)
                coverage = random.choice(self.coverages)
                year = random.choice(self.years)
                region = random.choice(self.regions)
                
                # Randomly use state name or code for natural language
                state_display = random.choice([state_name, state_code])
                
                # Use the SAME values for both NL and SQL
                nl = pattern["nl"].format(
                    metric=metric,
                    state=state_display,
                    state_code=state_code,
                    class_code=class_code,
                    coverage=coverage,
                    year=year,
                    region=region,
                )
                
                sql = pattern["sql"].format(
                    table=TABLE_NAME,
                    metric=metric,
                    state_code=state_code,
                    class_code=class_code,
                    coverage=coverage,
                    year=year,
                    region=region,
                )
                
                samples.append(NL2SQLSample(query=nl, sql=sql))
        
        return samples
    
    def generate_formula_samples(self, n: int = 500) -> List[NL2SQLSample]:
        """Generate samples specifically for formula calculations."""
        samples = []
        
        for formula in FORMULA_DEFINITIONS:
            for _ in range(n // len(FORMULA_DEFINITIONS)):
                # Generate various forms of formula questions
                state = random.choice(self.states)
                year = random.choice(self.years)
                class_code = random.choice(self.class_codes)
                
                # Question variations
                variations = [
                    f"What is the {formula.display_name.lower()} for {state}?",
                    f"Calculate {formula.display_name.lower()} by state",
                    f"Show {formula.display_name.lower()} for class code {class_code}",
                    f"What is the {formula.display_name.lower()} in {year}?",
                    f"Calculate the {formula.display_name.lower()} for {state} in {year}",
                ]
                
                nl = random.choice(variations)
                
                # Build SQL based on question
                if "by state" in nl:
                    sql = f"""SELECT state,
    {formula.sql_expression} AS {formula.name}
FROM {TABLE_NAME}
GROUP BY state
ORDER BY {formula.name} DESC
LIMIT 1000"""
                elif f"for {state}" in nl and str(year) in nl:
                    sql = f"""SELECT {formula.sql_expression} AS {formula.name}
FROM {TABLE_NAME}
WHERE state = '{state}' AND policy_year = {year}"""
                elif f"for {state}" in nl:
                    sql = f"""SELECT {formula.sql_expression} AS {formula.name}
FROM {TABLE_NAME}
WHERE state = '{state}'"""
                elif f"class code {class_code}" in nl:
                    sql = f"""SELECT {formula.sql_expression} AS {formula.name}
FROM {TABLE_NAME}
WHERE class_code = '{class_code}'"""
                elif str(year) in nl:
                    sql = f"""SELECT {formula.sql_expression} AS {formula.name}
FROM {TABLE_NAME}
WHERE policy_year = {year}"""
                else:
                    sql = f"""SELECT {formula.sql_expression} AS {formula.name}
FROM {TABLE_NAME}"""
                
                samples.append(NL2SQLSample(query=nl, sql=sql))
        
        return samples
    
    def generate_complex_samples(self, n: int = 200) -> List[NL2SQLSample]:
        """Generate complex multi-condition samples."""
        samples = []
        
        complex_patterns = [
            {
                "nl": "Show loss ratio for Workers Compensation in {state} for {year}, grouped by class code",
                "sql": """SELECT class_code,
    SUM(incurred_loss) / NULLIF(SUM(earned_premium), 0) AS loss_ratio
FROM {table}
WHERE coverage_type = 'Workers Compensation' 
    AND state = '{state}' 
    AND policy_year = {year}
GROUP BY class_code
ORDER BY loss_ratio DESC
LIMIT 1000""",
            },
            {
                "nl": "Compare claim frequency between {state1} and {state2} by coverage type",
                "sql": """SELECT state, coverage_type,
    SUM(claim_count) / NULLIF(SUM(exposure_units), 0) AS frequency
FROM {table}
WHERE state IN ('{state1}', '{state2}')
GROUP BY state, coverage_type
ORDER BY state, frequency DESC""",
            },
            {
                "nl": "Show top 5 class codes by pure premium in the {region} region for {year}",
                "sql": """SELECT class_code,
    SUM(incurred_loss) / NULLIF(SUM(exposure_units), 0) AS pure_premium
FROM {table}
WHERE region = '{region}' AND policy_year = {year}
GROUP BY class_code
ORDER BY pure_premium DESC
LIMIT 5""",
            },
            {
                "nl": "Calculate combined ratio for each region, excluding class code 8810",
                "sql": """SELECT region,
    (SUM(incurred_loss) + SUM(expense)) / NULLIF(SUM(earned_premium), 0) AS combined_ratio
FROM {table}
WHERE class_code != '8810'
GROUP BY region
ORDER BY combined_ratio DESC""",
            },
        ]
        
        for _ in range(n):
            pattern = random.choice(complex_patterns)
            
            # Pre-select all random values ONCE to ensure consistency between NL and SQL
            state1, state2 = random.sample(self.states, 2)
            state = random.choice(self.states)
            year = random.choice(self.years)
            region = random.choice(self.regions)
            
            # Use the SAME values for both NL and SQL
            nl = pattern["nl"].format(
                state=state,
                state1=state1,
                state2=state2,
                year=year,
                region=region,
            )
            
            sql = pattern["sql"].format(
                table=TABLE_NAME,
                state=state,
                state1=state1,
                state2=state2,
                year=year,
                region=region,
            )
            
            samples.append(NL2SQLSample(query=nl, sql=sql))
        
        return samples
    
    def generate_dataset(
        self,
        n_pattern: int = 50,
        n_formula: int = 1450,
        n_complex: int = 1600,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Generate complete NL2SQL training dataset.
        
        NL2SQL is the hardest task, ~4000 samples total.
        Distribution: harder subtasks get MORE samples.
        - pattern: 50 per template × ~19 templates = ~950 (basic - least samples)
        - formula: 1450 (medium - formula calculations)
        - complex: 1600 (hard - multi-condition queries, most samples)
        
        Args:
            n_pattern: Samples per pattern template (default 50)
            n_formula: Formula-specific samples (default 1450)
            n_complex: Complex multi-condition samples (default 1600)
            shuffle: Whether to shuffle
            
        Returns:
            List of training samples
        """
        samples = []
        
        samples.extend(self.generate_from_patterns(n_pattern))
        samples.extend(self.generate_formula_samples(n_formula))
        samples.extend(self.generate_complex_samples(n_complex))
        
        if shuffle:
            random.shuffle(samples)
        
        return [s.to_dict() for s in samples]
    
    def generate_train_test_split(
        self,
        n_pattern: int = 50,
        n_formula: int = 1450,
        n_complex: int = 1600,
        test_ratio: float = 0.15,
        shuffle: bool = True
    ) -> tuple:
        """
        Generate train and test datasets with specified split ratio.
        
        Args:
            n_pattern: Samples per pattern template
            n_formula: Formula-specific samples
            n_complex: Complex multi-condition samples
            test_ratio: Ratio of test samples (default 15%)
            shuffle: Whether to shuffle the datasets
            
        Returns:
            Tuple of (train_samples, test_samples) in dict format
        """
        # Generate all samples by category
        pattern_samples = self.generate_from_patterns(n_pattern)
        formula_samples = self.generate_formula_samples(n_formula)
        complex_samples = self.generate_complex_samples(n_complex)
        
        if shuffle:
            random.shuffle(pattern_samples)
            random.shuffle(formula_samples)
            random.shuffle(complex_samples)
        
        # Calculate split indices for each category
        def split_samples(samples, ratio):
            n_test = int(len(samples) * ratio)
            return samples[:-n_test] if n_test > 0 else samples, samples[-n_test:] if n_test > 0 else []
        
        pattern_train, pattern_test = split_samples(pattern_samples, test_ratio)
        formula_train, formula_test = split_samples(formula_samples, test_ratio)
        complex_train, complex_test = split_samples(complex_samples, test_ratio)
        
        # Combine train and test samples
        train_samples = pattern_train + formula_train + complex_train
        test_samples = pattern_test + formula_test + complex_test
        
        if shuffle:
            random.shuffle(train_samples)
            random.shuffle(test_samples)
        
        return (
            [s.to_dict() for s in train_samples],
            [s.to_dict() for s in test_samples]
        )
    
    def save_dataset(self, filepath: str, **kwargs):
        """Save dataset to JSON file."""
        dataset = self.generate_dataset(**kwargs)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(dataset)} samples to {filepath}")
    
    def save_train_test_split(
        self,
        train_filepath: str,
        test_filepath: str,
        n_pattern: int = 50,
        n_formula: int = 1450,
        n_complex: int = 1600,
        test_ratio: float = 0.15,
        **kwargs
    ):
        """
        Save train and test datasets to separate files.
        
        Args:
            train_filepath: Output path for training data
            test_filepath: Output path for test data
            n_pattern: Samples per pattern template
            n_formula: Formula-specific samples
            n_complex: Complex multi-condition samples
            test_ratio: Ratio of test samples (default 15%)
        """
        train_data, test_data = self.generate_train_test_split(
            n_pattern=n_pattern,
            n_formula=n_formula,
            n_complex=n_complex,
            test_ratio=test_ratio,
            **kwargs
        )
        
        # Save training data
        with open(train_filepath, 'w', encoding='utf-8') as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Save test data
        with open(test_filepath, 'w', encoding='utf-8') as f:
            for sample in test_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(train_data)} training samples to {train_filepath}")
        print(f"Saved {len(test_data)} test samples to {test_filepath}")
        print(f"  - Train/Test ratio: {100*(1-test_ratio):.0f}%/{100*test_ratio:.0f}%")

    def save_train_test_split_with_ground_truth(
        self,
        train_filepath: str,
        test_filepath: str,
        dataframe_filepath: str,
        n_pattern: int = 50,
        n_formula: int = 1450,
        n_complex: int = 1600,
        test_ratio: float = 0.15,
        dataframe_rows: int = 5000,
        seed: int = 42,
        **kwargs
    ):
        """
        Save train/test datasets and compute ground truth for NL2SQL tests.
        
        Generates a synthetic DataFrame and uses it to execute the expected SQL
        for each test sample, storing the results as ground truth.
        """
        train_data, test_data = self.generate_train_test_split(
            n_pattern=n_pattern,
            n_formula=n_formula,
            n_complex=n_complex,
            test_ratio=test_ratio,
            **kwargs
        )
        
        # Generate and save DataFrame
        df = self.generate_dataframe(n_rows=dataframe_rows, seed=seed)
        df.to_csv(dataframe_filepath, index=False)
        
        # Compute ground truth using DataFrame executor
        executor = SQLExecutor(use_dataframe=True, dataframe=df)
        for sample in test_data:
            expected_sql = sample.get("output", "")
            result = executor.execute(expected_sql)
            sample["ground_truth"] = {
                "success": result.success,
                "row_count": result.row_count,
                "data": self._serialize_result_data(result.data) if result.success else None,
                "error": result.error,
            }
        
        # Save training data
        with open(train_filepath, 'w', encoding='utf-8') as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Save test data with ground truth
        with open(test_filepath, 'w', encoding='utf-8') as f:
            for sample in test_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(train_data)} training samples to {train_filepath}")
        print(f"Saved {len(test_data)} test samples to {test_filepath}")
        print(f"Saved synthetic dataframe to {dataframe_filepath}")
        print(f"  - Train/Test ratio: {100*(1-test_ratio):.0f}%/{100*test_ratio:.0f}%")


if __name__ == "__main__":
    generator = NL2SQLDataGenerator()
    generator.save_dataset(
        "nl2sql_train.json",
        n_pattern=950,       # basic - ~950 samples (50 × 19 patterns)
        n_formula=1450,     # medium - formula calculations
        n_complex=1600      # hard - most samples for hardest subtask
    )
    # Total: ~4000 samples (harder subtasks get more samples)
