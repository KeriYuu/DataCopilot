"""
Insurance domain terminology for training data generation.

This module defines insurance-specific terms, their definitions, 
and mappings for entity extraction training.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class TermCategory(Enum):
    """Categories of insurance terminology."""
    COVERAGE = "coverage"        # Types of insurance coverage
    CLASS_CODE = "class_code"    # Industry classification codes
    METRIC = "metric"            # Financial/actuarial metrics
    GEOGRAPHY = "geography"      # Geographic terms
    TIME = "time"                # Time-related terms
    ENTITY = "entity"            # Policy/claim entities


@dataclass
class Term:
    """Definition of an insurance term."""
    name: str                    # Canonical name
    category: TermCategory
    definition: str              # Plain English definition
    schema_field: Optional[str]  # Corresponding database field
    aliases: List[str]           # Alternative names/abbreviations
    examples: List[str]          # Example values


# Coverage type definitions
COVERAGE_TERMS: List[Term] = [
    Term(
        name="Workers Compensation",
        category=TermCategory.COVERAGE,
        definition="Insurance providing wage replacement and medical benefits to employees injured in the course of employment",
        schema_field="coverage_type",
        aliases=["WC", "Workers Comp", "Work Comp", "Workman's Comp"],
        examples=["Workers Compensation"]
    ),
    Term(
        name="General Liability",
        category=TermCategory.COVERAGE,
        definition="Insurance protecting businesses against claims of bodily injury, property damage, and personal injury",
        schema_field="coverage_type",
        aliases=["GL", "CGL", "Commercial General Liability", "Liability"],
        examples=["General Liability"]
    ),
    Term(
        name="Commercial Auto",
        category=TermCategory.COVERAGE,
        definition="Insurance covering vehicles used for business purposes",
        schema_field="coverage_type",
        aliases=["Auto", "Business Auto", "Commercial Vehicle", "Fleet"],
        examples=["Commercial Auto"]
    ),
    Term(
        name="Commercial Property",
        category=TermCategory.COVERAGE,
        definition="Insurance covering business property against damage or loss",
        schema_field="coverage_type",
        aliases=["Property", "BOP", "Business Property"],
        examples=["Commercial Property"]
    ),
    Term(
        name="Professional Liability",
        category=TermCategory.COVERAGE,
        definition="Insurance protecting professionals against negligence claims",
        schema_field="coverage_type",
        aliases=["E&O", "Errors and Omissions", "Malpractice", "Professional Indemnity"],
        examples=["Professional Liability"]
    ),
]


# Class code definitions (NCCI codes for Workers Comp)
CLASS_CODE_TERMS: List[Term] = [
    Term(
        name="8810",
        category=TermCategory.CLASS_CODE,
        definition="Clerical Office Employees - office workers performing clerical duties",
        schema_field="class_code",
        aliases=["clerical", "office workers", "office employees"],
        examples=["8810"]
    ),
    Term(
        name="8742",
        category=TermCategory.CLASS_CODE,
        definition="Salespersons - Outside - sales personnel working outside the office",
        schema_field="class_code",
        aliases=["outside sales", "field sales", "sales reps"],
        examples=["8742"]
    ),
    Term(
        name="5506",
        category=TermCategory.CLASS_CODE,
        definition="Street or Road Construction - highway and road construction workers",
        schema_field="class_code",
        aliases=["road construction", "highway construction", "street work"],
        examples=["5506"]
    ),
    Term(
        name="8820",
        category=TermCategory.CLASS_CODE,
        definition="Attorneys - All Employees & Clerical - law firm employees",
        schema_field="class_code",
        aliases=["attorneys", "lawyers", "law firm", "legal"],
        examples=["8820"]
    ),
    Term(
        name="8017",
        category=TermCategory.CLASS_CODE,
        definition="Store - Retail NOC - retail store employees",
        schema_field="class_code",
        aliases=["retail", "store", "retail store"],
        examples=["8017"]
    ),
    Term(
        name="9015",
        category=TermCategory.CLASS_CODE,
        definition="Building Operation - Commercial - building maintenance and operation",
        schema_field="class_code",
        aliases=["building operations", "janitorial", "maintenance"],
        examples=["9015"]
    ),
    Term(
        name="5183",
        category=TermCategory.CLASS_CODE,
        definition="Plumbing NOC & Drivers - plumbing contractors",
        schema_field="class_code",
        aliases=["plumbing", "plumbers"],
        examples=["5183"]
    ),
    Term(
        name="5190",
        category=TermCategory.CLASS_CODE,
        definition="Electrical Wiring - Within Buildings & Drivers - electrical contractors",
        schema_field="class_code",
        aliases=["electrical", "electricians", "electrical wiring"],
        examples=["5190"]
    ),
]


# Metric definitions
METRIC_TERMS: List[Term] = [
    Term(
        name="Written Premium",
        category=TermCategory.METRIC,
        definition="The total premium charged for policies written during a period",
        schema_field="written_premium",
        aliases=["WP", "gross written premium", "GWP", "premium written"],
        examples=["$1,000,000 written premium"]
    ),
    Term(
        name="Earned Premium",
        category=TermCategory.METRIC,
        definition="The portion of written premium that has been 'earned' based on time elapsed",
        schema_field="earned_premium",
        aliases=["EP", "premium earned"],
        examples=["$800,000 earned premium"]
    ),
    Term(
        name="Incurred Loss",
        category=TermCategory.METRIC,
        definition="Total losses including paid amounts and reserves for unpaid claims",
        schema_field="incurred_loss",
        aliases=["incurred", "total loss", "losses incurred", "loss"],
        examples=["$500,000 incurred loss"]
    ),
    Term(
        name="Paid Loss",
        category=TermCategory.METRIC,
        definition="Actual cash payments made to settle claims",
        schema_field="paid_loss",
        aliases=["paid", "payments", "claim payments"],
        examples=["$300,000 paid loss"]
    ),
    Term(
        name="Reserved Loss",
        category=TermCategory.METRIC,
        definition="Estimated amounts set aside for unpaid/outstanding claims",
        schema_field="reserved_loss",
        aliases=["reserves", "outstanding", "IBNR", "case reserves"],
        examples=["$200,000 in reserves"]
    ),
    Term(
        name="Exposure Units",
        category=TermCategory.METRIC,
        definition="Units of measure for risk exposure (e.g., payroll in $100s for WC)",
        schema_field="exposure_units",
        aliases=["exposure", "units", "payroll exposure"],
        examples=["10,000 exposure units"]
    ),
    Term(
        name="Claim Count",
        category=TermCategory.METRIC,
        definition="Number of claims filed",
        schema_field="claim_count",
        aliases=["claims", "number of claims", "claim volume"],
        examples=["50 claims"]
    ),
]


# Geographic terminology
GEOGRAPHY_TERMS: List[Term] = [
    Term(
        name="West",
        category=TermCategory.GEOGRAPHY,
        definition="Western United States region",
        schema_field="region",
        aliases=["West Coast", "Western", "Pacific"],
        examples=["CA", "WA", "OR", "NV", "AZ"]
    ),
    Term(
        name="Northeast",
        category=TermCategory.GEOGRAPHY,
        definition="Northeastern United States region",
        schema_field="region",
        aliases=["East Coast", "Eastern", "New England"],
        examples=["NY", "NJ", "PA", "MA", "CT"]
    ),
    Term(
        name="Midwest",
        category=TermCategory.GEOGRAPHY,
        definition="Midwestern United States region",
        schema_field="region",
        aliases=["Central", "Great Lakes"],
        examples=["IL", "OH", "MI", "IN", "WI"]
    ),
    Term(
        name="South",
        category=TermCategory.GEOGRAPHY,
        definition="Southern United States region",
        schema_field="region",
        aliases=["Southeast", "Southern", "Sunbelt"],
        examples=["TX", "FL", "GA", "NC", "TN"]
    ),
]


# State abbreviation mappings
STATE_MAPPINGS: Dict[str, str] = {
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
    "new jersey": "NJ",
    "virginia": "VA",
    "washington": "WA",
    "arizona": "AZ",
    "massachusetts": "MA",
    "tennessee": "TN",
    "indiana": "IN",
    "missouri": "MO",
    "maryland": "MD",
    "wisconsin": "WI",
    "colorado": "CO",
    "minnesota": "MN",
    "south carolina": "SC",
    "alabama": "AL",
    "louisiana": "LA",
    "kentucky": "KY",
    "oregon": "OR",
    "oklahoma": "OK",
    "connecticut": "CT",
    "utah": "UT",
    "nevada": "NV",
    "arkansas": "AR",
    "iowa": "IA",
    "mississippi": "MS",
    "kansas": "KS",
    "new mexico": "NM",
    "nebraska": "NE",
    "west virginia": "WV",
    "idaho": "ID",
    "hawaii": "HI",
    "new hampshire": "NH",
    "maine": "ME",
    "montana": "MT",
    "rhode island": "RI",
    "delaware": "DE",
    "south dakota": "SD",
    "north dakota": "ND",
    "alaska": "AK",
    "vermont": "VT",
    "wyoming": "WY",
}

# Colloquial geographic mappings
COLLOQUIAL_GEO_MAPPINGS: Dict[str, Dict[str, str]] = {
    "socal": {"state": "CA", "region": "West"},
    "southern california": {"state": "CA", "region": "West"},
    "norcal": {"state": "CA", "region": "West"},
    "northern california": {"state": "CA", "region": "West"},
    "bay area": {"state": "CA", "region": "West"},
    "silicon valley": {"state": "CA", "region": "West"},
    "nyc": {"state": "NY", "region": "Northeast"},
    "new york city": {"state": "NY", "region": "Northeast"},
    "tri-state": {"region": "Northeast"},
    "dfw": {"state": "TX", "region": "South"},
    "dallas fort worth": {"state": "TX", "region": "South"},
    "chicagoland": {"state": "IL", "region": "Midwest"},
}


# All terminology combined
TERMINOLOGY_DEFINITIONS: List[Term] = (
    COVERAGE_TERMS + 
    CLASS_CODE_TERMS + 
    METRIC_TERMS + 
    GEOGRAPHY_TERMS
)


class InsuranceTerminology:
    """
    Insurance terminology manager for training data generation.
    
    Used for:
    1. Entity extraction training
    2. Keyword normalization
    3. Query understanding
    """
    
    def __init__(self):
        self._terms = {t.name: t for t in TERMINOLOGY_DEFINITIONS}
        self._alias_map = self._build_alias_map()
        self._state_map = STATE_MAPPINGS
        self._geo_map = COLLOQUIAL_GEO_MAPPINGS
    
    def _build_alias_map(self) -> Dict[str, str]:
        """Build mapping from aliases to term names."""
        alias_map = {}
        for term in TERMINOLOGY_DEFINITIONS:
            alias_map[term.name.lower()] = term.name
            for alias in term.aliases:
                alias_map[alias.lower()] = term.name
        return alias_map
    
    def get_term(self, name: str) -> Optional[Term]:
        """Get term by name or alias."""
        normalized = name.lower().strip()
        term_name = self._alias_map.get(normalized)
        if term_name:
            return self._terms.get(term_name)
        return None
    
    def normalize_state(self, state_text: str) -> Optional[str]:
        """Normalize state name to 2-letter code."""
        normalized = state_text.lower().strip()
        
        # Already a code
        if len(normalized) == 2 and normalized.upper() in STATE_MAPPINGS.values():
            return normalized.upper()
        
        # Full name
        if normalized in self._state_map:
            return self._state_map[normalized]
        
        return None
    
    def normalize_geography(self, geo_text: str) -> Dict[str, str]:
        """Normalize colloquial geography to schema fields."""
        normalized = geo_text.lower().strip()
        
        # Check colloquial mappings first
        if normalized in self._geo_map:
            return self._geo_map[normalized]
        
        # Check state
        state = self.normalize_state(geo_text)
        if state:
            return {"state": state}
        
        return {}
    
    def extract_entities(self, query: str) -> Dict[str, str]:
        """
        Extract entities from a natural language query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of {schema_field: value}
        """
        entities = {}
        query_lower = query.lower()
        
        # Check for terms
        for alias, term_name in self._alias_map.items():
            if alias in query_lower:
                term = self._terms.get(term_name)
                if term and term.schema_field:
                    # Use canonical name as value
                    entities[term.schema_field] = term.name
        
        # Check for state references
        for state_name, state_code in self._state_map.items():
            if state_name in query_lower:
                entities["state"] = state_code
                break
        
        # Check for 2-letter state codes
        import re
        state_codes = re.findall(r'\b([A-Z]{2})\b', query)
        for code in state_codes:
            if code in STATE_MAPPINGS.values():
                entities["state"] = code
                break
        
        # Check colloquial geography
        for colloquial, mapping in self._geo_map.items():
            if colloquial in query_lower:
                entities.update(mapping)
        
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', query)
        if years:
            # Determine if policy_year or accident_year based on context
            if "accident" in query_lower or "occurrence" in query_lower:
                entities["accident_year"] = int(years[0])
            elif "policy" in query_lower or "effective" in query_lower:
                entities["policy_year"] = int(years[0])
            else:
                entities["policy_year"] = int(years[0])
        
        # Extract class codes
        class_codes = re.findall(r'\b(\d{4})\b', query)
        for code in class_codes:
            if code in [t.name for t in CLASS_CODE_TERMS]:
                entities["class_code"] = code
                break
        
        return entities
    
    def get_definition(self, name: str) -> Optional[str]:
        """Get definition for a term."""
        term = self.get_term(name)
        if term:
            return term.definition
        return None
    
    def get_all_terms(self) -> List[Term]:
        """Get all defined terms."""
        return TERMINOLOGY_DEFINITIONS
    
    def get_terms_by_category(self, category: TermCategory) -> List[Term]:
        """Get terms by category."""
        return [t for t in TERMINOLOGY_DEFINITIONS if t.category == category]
    
    def get_class_code_reference(self) -> Dict[str, str]:
        """Get class code reference dictionary."""
        return {t.name: t.definition for t in CLASS_CODE_TERMS}
