"""
Intent classification training data generator.

Generates training samples for the Intent LoRA adapter that classifies
queries into: DATA_QUERY or METADATA (simplified 2-class classification).
"""
import json
import random
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

from training.domain.formulas import InsuranceFormulas, FormulaCategory
from training.domain.terminology import InsuranceTerminology, TermCategory


class IntentType(str, Enum):
    """Simplified intent types for classification."""
    DATA_QUERY = "DATA_QUERY"
    METADATA = "METADATA"


@dataclass
class IntentSample:
    """A single intent classification training sample."""
    query: str
    intent: IntentType
    
    def to_dict(self) -> Dict:
        return {
            "instruction": "Classify the following insurance data query into one of two categories: DATA_QUERY (any database query including lookups, aggregations, calculations) or METADATA (definitions, system info, general questions). Respond with ONLY the category name.",
            "input": self.query,
            "output": self.intent.value
        }


# ==================== DATA_QUERY TEMPLATES ====================
# Merged from DIRECT_QUERY and STATISTICAL_AGGREGATION

# Simple lookup queries
DIRECT_QUERY_TEMPLATES = [
    "Show me policy {policy_id}",
    "Get details for claim {claim_id}",
    "What is the premium for policy {policy_id}?",
    "Show claim {claim_id} information",
    "Find policy ID {policy_id}",
    "Look up claim number {claim_id}",
    "Display the record for {policy_id}",
    "Retrieve policy {policy_id} details",
    "What are the details of claim {claim_id}?",
    "Show me the loss for claim {claim_id}",
    "Get the exposure units for {policy_id}",
    "What is the incurred loss for claim {claim_id}?",
]

# Aggregation and calculation queries
STATISTICAL_TEMPLATES = [
    "What is the {metric} by {dimension}?",
    "Calculate the average {metric} for {state}",
    "Show total {metric} grouped by {dimension}",
    "What is the {metric} for class code {class_code}?",
    "Calculate {formula} by state",
    "What is the {formula} for {coverage} in {state}?",
    "Show me the top 10 {dimension}s by {metric}",
    "Compare {metric} across {dimension}s",
    "What is the sum of {metric} for {year}?",
    "Calculate the {formula} trend from {year} to {year2}",
    "Which {dimension} has the highest {metric}?",
    "What is the average {metric} per {dimension}?",
    "Show {formula} breakdown by {dimension}",
    "Rank {dimension}s by {metric} for {year}",
    "What percentage of {metric} comes from {state}?",
    "Compare {formula} between {state} and {state2}",
    "What is the year-over-year change in {metric}?",
    "Show monthly {metric} for {year}",
    "Calculate {metric} growth rate by {dimension}",
    "What is the {metric} variance across {dimension}s?",
]

# All DATA_QUERY templates combined
DATA_QUERY_TEMPLATES = DIRECT_QUERY_TEMPLATES + STATISTICAL_TEMPLATES


# ==================== METADATA TEMPLATES ====================

METADATA_TEMPLATES = [
    "What is {term}?",
    "Define {term}",
    "What does {term} mean?",
    "Explain {term}",
    "What is class code {class_code}?",
    "What columns are available in the database?",
    "How is {formula} calculated?",
    "What is the definition of {term}?",
    "What does {class_code} cover?",
    "What types of coverage do you have data for?",
    "What states are included in the data?",
    "How do you calculate {formula}?",
    "What is the formula for {formula}?",
    "What metrics can I query?",
    "What dimensions can I group by?",
    "Explain the difference between {term} and {term2}",
    "What does the {column} column represent?",
    "What are the available filters?",
    "What time period does the data cover?",
    "What is the data source?",
]


class IntentDataGenerator:
    """
    Generator for intent classification training data.
    
    Creates balanced training samples for the two intent categories:
    - DATA_QUERY: All database queries (lookups + aggregations + calculations)
    - METADATA: Definitions, system info, general questions
    """
    
    def __init__(self):
        self.formulas = InsuranceFormulas()
        self.terminology = InsuranceTerminology()
        
        # Sample values for templates
        self.states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
        self.years = ["2020", "2021", "2022", "2023", "2024"]
        self.class_codes = ["8810", "8742", "5506", "8820", "8017", "9015", "8832"]
        self.coverages = ["Workers Compensation", "General Liability", "Commercial Auto"]
        self.dimensions = ["state", "class_code", "coverage_type", "policy_year", "region"]
        self.metrics = ["written_premium", "incurred_loss", "claim_count", "exposure_units", "earned_premium"]
        self.columns = ["policy_id", "claim_id", "state", "class_code", "premium", "loss"]
    
    def generate_data_query_samples(self, n: int = 1500) -> List[IntentSample]:
        """
        Generate DATA_QUERY samples (merged direct + statistical).
        
        Args:
            n: Number of samples to generate
            
        Returns:
            List of IntentSample objects
        """
        samples = []
        
        formula_names = [f.display_name for f in self.formulas.get_all_formulas()]
        
        for _ in range(n):
            template = random.choice(DATA_QUERY_TEMPLATES)
            
            # Fill in template variables
            query = template.format(
                policy_id=f"POL-{random.randint(2020, 2024)}-{random.randint(100000, 999999)}",
                claim_id=f"CLM-{random.randint(2020, 2024)}-{random.randint(1000, 9999)}",
                metric=random.choice(self.metrics),
                dimension=random.choice(self.dimensions),
                state=random.choice(self.states),
                state2=random.choice(self.states),
                class_code=random.choice(self.class_codes),
                formula=random.choice(formula_names),
                coverage=random.choice(self.coverages),
                year=random.choice(self.years),
                year2=str(int(random.choice(self.years)) + 1),
            )
            
            samples.append(IntentSample(
                query=query,
                intent=IntentType.DATA_QUERY,
            ))
        
        return samples
    
    def generate_metadata_samples(self, n: int = 500) -> List[IntentSample]:
        """
        Generate METADATA samples.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            List of IntentSample objects
        """
        samples = []
        
        terms = [t.name for t in self.terminology.get_all_terms()]
        formula_names = [f.display_name for f in self.formulas.get_all_formulas()]
        
        for _ in range(n):
            template = random.choice(METADATA_TEMPLATES)
            
            query = template.format(
                term=random.choice(terms) if terms else "loss ratio",
                term2=random.choice(terms) if terms else "pure premium",
                class_code=random.choice(self.class_codes),
                formula=random.choice(formula_names),
                column=random.choice(self.columns),
            )
            
            samples.append(IntentSample(
                query=query,
                intent=IntentType.METADATA,
            ))
        
        return samples
    
    def generate_dataset(
        self, 
        n_data_query: int = 550,
        n_metadata: int = 250,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Generate complete training dataset.
        
        Intent classification is a simple task, ~800 samples is sufficient.
        Distribution based on expected query frequency:
        - DATA_QUERY: 550 (~70%) - more common in production
        - METADATA: 250 (~30%) - less frequent
        
        Args:
            n_data_query: Number of DATA_QUERY samples
            n_metadata: Number of METADATA samples
            shuffle: Whether to shuffle the dataset
            
        Returns:
            List of training samples in dict format
        """
        samples = []
        
        samples.extend(self.generate_data_query_samples(n_data_query))
        samples.extend(self.generate_metadata_samples(n_metadata))
        
        if shuffle:
            random.shuffle(samples)
        
        return [s.to_dict() for s in samples]
    
    def generate_train_test_split(
        self,
        n_data_query: int = 550,
        n_metadata: int = 250,
        test_ratio: float = 0.15,
        shuffle: bool = True
    ) -> tuple:
        """
        Generate train and test datasets with specified split ratio.
        
        Args:
            n_data_query: Total number of DATA_QUERY samples
            n_metadata: Total number of METADATA samples
            test_ratio: Ratio of test samples (default 15%)
            shuffle: Whether to shuffle the datasets
            
        Returns:
            Tuple of (train_samples, test_samples) in dict format
        """
        # Calculate train/test counts for each category
        n_data_query_test = int(n_data_query * test_ratio)
        n_data_query_train = n_data_query - n_data_query_test
        
        n_metadata_test = int(n_metadata * test_ratio)
        n_metadata_train = n_metadata - n_metadata_test
        
        # Generate samples for each split
        data_query_samples = self.generate_data_query_samples(n_data_query)
        metadata_samples = self.generate_metadata_samples(n_metadata)
        
        if shuffle:
            random.shuffle(data_query_samples)
            random.shuffle(metadata_samples)
        
        # Split into train and test
        train_samples = (
            data_query_samples[:n_data_query_train] +
            metadata_samples[:n_metadata_train]
        )
        test_samples = (
            data_query_samples[n_data_query_train:] +
            metadata_samples[n_metadata_train:]
        )
        
        if shuffle:
            random.shuffle(train_samples)
            random.shuffle(test_samples)
        
        return (
            [s.to_dict() for s in train_samples],
            [s.to_dict() for s in test_samples]
        )
    
    def save_dataset(
        self, 
        filepath: str, 
        n_data_query: int = 550,
        n_metadata: int = 250,
        **kwargs
    ):
        """
        Save dataset to JSON file.
        
        Args:
            filepath: Output file path
            n_data_query: Number of DATA_QUERY samples (default 550)
            n_metadata: Number of METADATA samples (default 250)
        """
        dataset = self.generate_dataset(
            n_data_query=n_data_query,
            n_metadata=n_metadata,
            **kwargs
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(dataset)} samples to {filepath}")
        print(f"  - DATA_QUERY: {n_data_query}")
        print(f"  - METADATA: {n_metadata}")
    
    def save_train_test_split(
        self,
        train_filepath: str,
        test_filepath: str,
        n_data_query: int = 550,
        n_metadata: int = 250,
        test_ratio: float = 0.15,
        **kwargs
    ):
        """
        Save train and test datasets to separate files.
        
        Args:
            train_filepath: Output path for training data
            test_filepath: Output path for test data
            n_data_query: Total number of DATA_QUERY samples
            n_metadata: Total number of METADATA samples
            test_ratio: Ratio of test samples (default 15%)
        """
        train_data, test_data = self.generate_train_test_split(
            n_data_query=n_data_query,
            n_metadata=n_metadata,
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


if __name__ == "__main__":
    generator = IntentDataGenerator()
    generator.save_dataset(
        "intent_train.json",
        n_data_query=550,
        n_metadata=250
    )
    # Total: 800 samples (simple classification task)
