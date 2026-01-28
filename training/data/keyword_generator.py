"""
Keyword/Entity extraction training data generator.

Generates training samples for extracting and normalizing entities
from natural language queries (state names, class codes, coverage types, etc.).
"""
import json
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from training.domain.terminology import (
    InsuranceTerminology, 
    STATE_MAPPINGS, 
    COLLOQUIAL_GEO_MAPPINGS,
    COVERAGE_TERMS,
    CLASS_CODE_TERMS,
    METRIC_TERMS,
)


@dataclass
class KeywordSample:
    """A single keyword extraction training sample."""
    query: str
    entities: Dict[str, str]
    rewritten_query: str
    
    def to_dict(self) -> Dict:
        output = {
            "entities": self.entities,
            "rewritten_query": self.rewritten_query
        }
        
        return {
            "instruction": """Extract and normalize entities from the following insurance data query.
Map colloquial terms to schema values:
- State names → 2-letter codes (California → CA)
- Coverage nicknames → full names (WC → Workers Compensation)
- Geographic slang → proper regions (SoCal → CA/West)

Output JSON with 'entities' (field:value pairs) and 'rewritten_query' (normalized query).""",
            "input": self.query,
            "output": json.dumps(output, ensure_ascii=False)
        }


# Query templates with entity variations
KEYWORD_TEMPLATES = [
    # State variations
    {
        "query": "Show me data for {state_informal}",
        "entities": {"state": "{state_code}"},
        "rewrite": "Show me data for {state_code}",
    },
    {
        "query": "What is the loss ratio in {state_informal}?",
        "entities": {"state": "{state_code}"},
        "rewrite": "What is the loss ratio in {state_code}?",
    },
    {
        "query": "Calculate premium for {state_informal} in {year}",
        "entities": {"state": "{state_code}", "policy_year": "{year}"},
        "rewrite": "Calculate premium for {state_code} in {year}",
    },
    
    # Coverage variations
    {
        "query": "Show {coverage_informal} stats",
        "entities": {"coverage_type": "{coverage_formal}"},
        "rewrite": "Show {coverage_formal} stats",
    },
    {
        "query": "What is the frequency for {coverage_informal} in {state_informal}?",
        "entities": {"coverage_type": "{coverage_formal}", "state": "{state_code}"},
        "rewrite": "What is the frequency for {coverage_formal} in {state_code}?",
    },
    
    # Geographic slang
    {
        "query": "Show stats for {geo_slang}",
        "entities": "{geo_entities}",
        "rewrite": "Show stats for {geo_formal}",
    },
    {
        "query": "Calculate loss ratio in {geo_slang} for {coverage_informal}",
        "entities": "{combined_entities}",
        "rewrite": "Calculate loss ratio in {geo_formal} for {coverage_formal}",
    },
    
    # Class codes
    {
        "query": "Show data for {class_informal}",
        "entities": {"class_code": "{class_code}"},
        "rewrite": "Show data for class code {class_code}",
    },
    {
        "query": "What is the severity for {class_informal} workers?",
        "entities": {"class_code": "{class_code}"},
        "rewrite": "What is the severity for class code {class_code}?",
    },
    
    # Combined
    {
        "query": "Show {coverage_informal} loss ratio for {class_informal} in {state_informal}",
        "entities": {
            "coverage_type": "{coverage_formal}",
            "class_code": "{class_code}",
            "state": "{state_code}"
        },
        "rewrite": "Show {coverage_formal} loss ratio for class code {class_code} in {state_code}",
    },
    
    # Year extraction
    {
        "query": "Premium for {state_informal} in {year}",
        "entities": {"state": "{state_code}", "policy_year": "{year}"},
        "rewrite": "Premium for {state_code} in {year}",
    },
    {
        "query": "Compare {year1} to {year2} for {state_informal}",
        "entities": {"state": "{state_code}", "policy_year": "{year1}"},
        "rewrite": "Compare {year1} to {year2} for {state_code}",
    },
]


# Coverage informal to formal mappings
COVERAGE_MAPPINGS = {
    "WC": "Workers Compensation",
    "Workers Comp": "Workers Compensation",
    "Work Comp": "Workers Compensation",
    "Workman's Comp": "Workers Compensation",
    "GL": "General Liability",
    "CGL": "General Liability",
    "Liability": "General Liability",
    "Auto": "Commercial Auto",
    "Business Auto": "Commercial Auto",
    "Fleet": "Commercial Auto",
    "Property": "Commercial Property",
    "BOP": "Commercial Property",
    "E&O": "Professional Liability",
    "Errors and Omissions": "Professional Liability",
}


# Class code informal references
CLASS_CODE_INFORMAL = {
    "8810": ["clerical", "office workers", "desk workers", "administrative"],
    "8742": ["outside sales", "field sales", "sales reps", "traveling sales"],
    "5506": ["road construction", "highway workers", "road crews"],
    "8820": ["attorneys", "lawyers", "law firm", "legal staff"],
    "8017": ["retail workers", "store employees", "retail staff"],
    "9015": ["janitors", "building maintenance", "custodial"],
    "5183": ["plumbers", "plumbing contractors"],
    "5190": ["electricians", "electrical contractors"],
}


class KeywordDataGenerator:
    """
    Generator for keyword/entity extraction training data.
    
    Creates samples that teach the model to:
    1. Extract entities from natural language
    2. Normalize informal terms to schema values
    3. Handle geographic slang and abbreviations
    """
    
    def __init__(self):
        self.terminology = InsuranceTerminology()
        
        # State mappings
        self.state_mappings = STATE_MAPPINGS
        self.geo_slang = COLLOQUIAL_GEO_MAPPINGS
        
        # Build reverse lookups
        self.state_formal_to_code = {v.title(): k for k, v in STATE_MAPPINGS.items()}
        
        self.years = [2019, 2020, 2021, 2022, 2023]
    
    def generate_state_samples(self, n: int = 200) -> List[KeywordSample]:
        """Generate samples with state name variations."""
        samples = []
        
        for state_name, state_code in self.state_mappings.items():
            for _ in range(n // len(self.state_mappings)):
                # Variations: full name, code, title case
                state_informal = random.choice([
                    state_name.title(),
                    state_name.upper(),
                    state_name,
                ])
                
                year = random.choice(self.years)
                
                template = random.choice([
                    ("Show data for {state}", {"state": state_code}, f"Show data for {state_code}"),
                    ("Loss ratio in {state}", {"state": state_code}, f"Loss ratio in {state_code}"),
                    ("{state} premium for {year}", {"state": state_code, "policy_year": year}, f"{state_code} premium for {year}"),
                ])
                
                query = template[0].format(state=state_informal, year=year)
                rewrite = template[2].format(year=year)
                
                samples.append(KeywordSample(
                    query=query,
                    entities=template[1],
                    rewritten_query=rewrite
                ))
        
        return samples
    
    def generate_coverage_samples(self, n: int = 200) -> List[KeywordSample]:
        """Generate samples with coverage type variations."""
        samples = []
        
        for informal, formal in COVERAGE_MAPPINGS.items():
            for _ in range(n // len(COVERAGE_MAPPINGS)):
                state_code = random.choice(list(self.state_mappings.values())[:20])
                
                templates = [
                    (
                        f"Show {informal} data",
                        {"coverage_type": formal},
                        f"Show {formal} data"
                    ),
                    (
                        f"{informal} loss ratio",
                        {"coverage_type": formal},
                        f"{formal} loss ratio"
                    ),
                    (
                        f"Calculate {informal} frequency in {state_code}",
                        {"coverage_type": formal, "state": state_code},
                        f"Calculate {formal} frequency in {state_code}"
                    ),
                ]
                
                template = random.choice(templates)
                
                samples.append(KeywordSample(
                    query=template[0],
                    entities=template[1],
                    rewritten_query=template[2]
                ))
        
        return samples
    
    def generate_geo_slang_samples(self, n: int = 100) -> List[KeywordSample]:
        """Generate samples with geographic slang."""
        samples = []
        
        for slang, mapping in self.geo_slang.items():
            for _ in range(n // len(self.geo_slang)):
                formal = mapping.get("state", mapping.get("region", ""))
                
                templates = [
                    (
                        f"Show data for {slang}",
                        mapping,
                        f"Show data for {formal}"
                    ),
                    (
                        f"Loss ratio in {slang}",
                        mapping,
                        f"Loss ratio in {formal}"
                    ),
                    (
                        f"Calculate premium for {slang}",
                        mapping,
                        f"Calculate premium for {formal}"
                    ),
                ]
                
                template = random.choice(templates)
                
                samples.append(KeywordSample(
                    query=template[0],
                    entities=template[1],
                    rewritten_query=template[2]
                ))
        
        return samples
    
    def generate_class_code_samples(self, n: int = 150) -> List[KeywordSample]:
        """Generate samples with class code informal references."""
        samples = []
        
        for class_code, informal_list in CLASS_CODE_INFORMAL.items():
            for informal in informal_list:
                for _ in range(n // (len(CLASS_CODE_INFORMAL) * 3)):
                    state_code = random.choice(list(self.state_mappings.values())[:20])
                    
                    templates = [
                        (
                            f"Show data for {informal}",
                            {"class_code": class_code},
                            f"Show data for class code {class_code}"
                        ),
                        (
                            f"Loss ratio for {informal} in {state_code}",
                            {"class_code": class_code, "state": state_code},
                            f"Loss ratio for class code {class_code} in {state_code}"
                        ),
                        (
                            f"How many {informal} claims?",
                            {"class_code": class_code},
                            f"How many class code {class_code} claims?"
                        ),
                    ]
                    
                    template = random.choice(templates)
                    
                    samples.append(KeywordSample(
                        query=template[0],
                        entities=template[1],
                        rewritten_query=template[2]
                    ))
        
        return samples
    
    def generate_combined_samples(self, n: int = 200) -> List[KeywordSample]:
        """Generate samples with multiple entity types."""
        samples = []
        
        for _ in range(n):
            # Pick random values
            state_name = random.choice(list(self.state_mappings.keys()))
            state_code = self.state_mappings[state_name]
            coverage_informal = random.choice(list(COVERAGE_MAPPINGS.keys()))
            coverage_formal = COVERAGE_MAPPINGS[coverage_informal]
            class_code = random.choice(list(CLASS_CODE_INFORMAL.keys()))
            class_informal = random.choice(CLASS_CODE_INFORMAL[class_code])
            year = random.choice(self.years)
            
            templates = [
                (
                    f"Show {coverage_informal} loss ratio for {state_name.title()} in {year}",
                    {"coverage_type": coverage_formal, "state": state_code, "policy_year": year},
                    f"Show {coverage_formal} loss ratio for {state_code} in {year}"
                ),
                (
                    f"Calculate frequency for {class_informal} in {state_name.title()}",
                    {"class_code": class_code, "state": state_code},
                    f"Calculate frequency for class code {class_code} in {state_code}"
                ),
                (
                    f"{coverage_informal} severity for {class_informal} workers in {state_name.title()}",
                    {"coverage_type": coverage_formal, "class_code": class_code, "state": state_code},
                    f"{coverage_formal} severity for class code {class_code} in {state_code}"
                ),
            ]
            
            template = random.choice(templates)
            
            samples.append(KeywordSample(
                query=template[0],
                entities=template[1],
                rewritten_query=template[2]
            ))
        
        return samples
    
    def generate_dataset(
        self,
        n_state: int = 200,
        n_coverage: int = 250,
        n_geo: int = 350,
        n_class: int = 450,
        n_combined: int = 750,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Generate complete keyword extraction dataset.
        
        Keyword extraction is medium difficulty, ~2000 samples total.
        Distribution: harder tasks get MORE samples.
        - state: 200 (simple - direct state name mapping)
        - coverage: 250 (simple - coverage type abbreviations)
        - geo: 350 (medium - geographic slang requires context)
        - class_code: 450 (medium - occupation codes require domain knowledge)
        - combined: 750 (hard - multiple entities in one query)
        """
        samples = []
        
        samples.extend(self.generate_state_samples(n_state))
        samples.extend(self.generate_coverage_samples(n_coverage))
        samples.extend(self.generate_geo_slang_samples(n_geo))
        samples.extend(self.generate_class_code_samples(n_class))
        samples.extend(self.generate_combined_samples(n_combined))
        
        if shuffle:
            random.shuffle(samples)
        
        return [s.to_dict() for s in samples]
    
    def generate_train_test_split(
        self,
        n_state: int = 200,
        n_coverage: int = 250,
        n_geo: int = 350,
        n_class: int = 450,
        n_combined: int = 750,
        test_ratio: float = 0.15,
        shuffle: bool = True
    ) -> tuple:
        """
        Generate train and test datasets with specified split ratio.
        
        Args:
            n_state: Total number of state samples
            n_coverage: Total number of coverage samples
            n_geo: Total number of geo slang samples
            n_class: Total number of class code samples
            n_combined: Total number of combined samples
            test_ratio: Ratio of test samples (default 15%)
            shuffle: Whether to shuffle the datasets
            
        Returns:
            Tuple of (train_samples, test_samples) in dict format
        """
        # Generate all samples by category
        state_samples = self.generate_state_samples(n_state)
        coverage_samples = self.generate_coverage_samples(n_coverage)
        geo_samples = self.generate_geo_slang_samples(n_geo)
        class_samples = self.generate_class_code_samples(n_class)
        combined_samples = self.generate_combined_samples(n_combined)
        
        if shuffle:
            random.shuffle(state_samples)
            random.shuffle(coverage_samples)
            random.shuffle(geo_samples)
            random.shuffle(class_samples)
            random.shuffle(combined_samples)
        
        # Calculate split indices for each category
        def split_samples(samples, ratio):
            n_test = int(len(samples) * ratio)
            return samples[:-n_test] if n_test > 0 else samples, samples[-n_test:] if n_test > 0 else []
        
        state_train, state_test = split_samples(state_samples, test_ratio)
        coverage_train, coverage_test = split_samples(coverage_samples, test_ratio)
        geo_train, geo_test = split_samples(geo_samples, test_ratio)
        class_train, class_test = split_samples(class_samples, test_ratio)
        combined_train, combined_test = split_samples(combined_samples, test_ratio)
        
        # Combine train and test samples
        train_samples = state_train + coverage_train + geo_train + class_train + combined_train
        test_samples = state_test + coverage_test + geo_test + class_test + combined_test
        
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
        n_state: int = 200,
        n_coverage: int = 250,
        n_geo: int = 350,
        n_class: int = 450,
        n_combined: int = 750,
        test_ratio: float = 0.15,
        **kwargs
    ):
        """
        Save train and test datasets to separate files.
        
        Args:
            train_filepath: Output path for training data
            test_filepath: Output path for test data
            n_state: Total number of state samples
            n_coverage: Total number of coverage samples
            n_geo: Total number of geo slang samples
            n_class: Total number of class code samples
            n_combined: Total number of combined samples
            test_ratio: Ratio of test samples (default 15%)
        """
        train_data, test_data = self.generate_train_test_split(
            n_state=n_state,
            n_coverage=n_coverage,
            n_geo=n_geo,
            n_class=n_class,
            n_combined=n_combined,
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
    generator = KeywordDataGenerator()
    generator.save_dataset(
        "keyword_train.json",
        n_state=200,      # simple
        n_coverage=250,   # simple
        n_geo=350,        # medium
        n_class=450,      # medium
        n_combined=750    # hard - most samples
    )
    # Total: 2000 samples (harder subtasks get more samples)
