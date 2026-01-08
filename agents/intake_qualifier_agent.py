"""
GENESIS Intake Qualifier Agent (Agent 1)
===============================================================================
Validates incoming briefs have enough information to start video production.
First gate in the 9-phase pipeline.

Features:
- Rule-based validation for required fields
- Claude-powered extraction from freeform briefs (optional)
- Scoring: 0-40 REJECTED, 41-70 NEEDS_INFO, 71-100 QUALIFIED
- Industry detection and normalization
- Structured data extraction

Author: Barrios A2I
Version: 1.0.0 (GENESIS Standalone)
===============================================================================
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("genesis.intake_qualifier")


# =============================================================================
# ENUMS
# =============================================================================

class QualificationStatus(str, Enum):
    """Brief qualification status."""
    QUALIFIED = "qualified"       # Score 71-100: Ready for production
    NEEDS_INFO = "needs_info"     # Score 41-70: Can proceed with warnings
    REJECTED = "rejected"         # Score 0-40: Missing critical info


class FieldCategory(str, Enum):
    """Field importance categories."""
    CRITICAL = "critical"     # Must have (20 points each)
    IMPORTANT = "important"   # Should have (10 points each)
    OPTIONAL = "optional"     # Nice to have (5 points each)


# =============================================================================
# DATA MODELS
# =============================================================================

class QualificationRequest(BaseModel):
    """Request for brief qualification."""
    brief: Dict[str, Any] = Field(..., description="The brief to qualify")
    business_name: Optional[str] = Field(None, description="Business name if known")
    industry: Optional[str] = Field(None, description="Industry if known")
    strict: bool = Field(default=False, description="Strict mode rejects NEEDS_INFO")


class ExtractedData(BaseModel):
    """Structured data extracted from brief."""
    business_name: Optional[str] = None
    industry: Optional[str] = None
    product_service: Optional[str] = None
    target_audience: Optional[str] = None
    campaign_goals: List[str] = Field(default_factory=list)
    tone: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    call_to_action: Optional[str] = None
    duration_preference: Optional[int] = None
    budget_range: Optional[str] = None


class QualificationResult(BaseModel):
    """Result from brief qualification."""
    status: QualificationStatus
    score: int = Field(..., ge=0, le=100)
    missing_fields: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    extracted_data: ExtractedData = Field(default_factory=ExtractedData)
    warnings: List[str] = Field(default_factory=list)
    source: str = "rule_based"


# =============================================================================
# FIELD DEFINITIONS
# =============================================================================

# Required/optional fields with their weights
FIELD_DEFINITIONS = {
    # CRITICAL (20 points each) - Must have for production
    "business_name": {
        "category": FieldCategory.CRITICAL,
        "weight": 20,
        "aliases": ["company", "brand", "name", "company_name", "client_name"],
        "description": "Business or company name"
    },
    "industry": {
        "category": FieldCategory.CRITICAL,
        "weight": 15,
        "aliases": ["vertical", "sector", "niche", "business_type"],
        "description": "Industry/vertical"
    },
    "product_service": {
        "category": FieldCategory.CRITICAL,
        "weight": 20,
        "aliases": ["product", "service", "offering", "description", "what_you_do", "primary_offering"],
        "description": "Product or service description"
    },
    "target_audience": {
        "category": FieldCategory.CRITICAL,
        "weight": 15,
        "aliases": ["audience", "target", "demographic", "customer", "buyer", "target_demographic"],
        "description": "Target audience"
    },
    "campaign_goals": {
        "category": FieldCategory.IMPORTANT,
        "weight": 10,
        "aliases": ["goals", "objectives", "purpose", "video_goal"],
        "description": "Campaign goals/objectives"
    },

    # IMPORTANT (10 points each) - Should have
    "call_to_action": {
        "category": FieldCategory.IMPORTANT,
        "weight": 10,
        "aliases": ["cta", "action", "next_step"],
        "description": "Call to action"
    },
    "tone": {
        "category": FieldCategory.IMPORTANT,
        "weight": 10,
        "aliases": ["style", "voice", "mood", "brand_voice"],
        "description": "Tone/style"
    },

    # OPTIONAL (5 points each) - Nice to have
    "platforms": {
        "category": FieldCategory.OPTIONAL,
        "weight": 5,
        "aliases": ["platform", "channels", "where"],
        "description": "Target platforms"
    },
    "pain_points": {
        "category": FieldCategory.OPTIONAL,
        "weight": 5,
        "aliases": ["problems", "challenges", "issues"],
        "description": "Customer pain points"
    },
    "budget_range": {
        "category": FieldCategory.OPTIONAL,
        "weight": 5,
        "aliases": ["budget", "investment", "spend"],
        "description": "Budget range"
    },
}

# Industry mapping for normalization
INDUSTRY_MAP = {
    # Technology
    "software": "technology", "saas": "technology", "tech": "technology",
    "it": "technology", "app": "technology", "startup": "technology",
    # Healthcare
    "medical": "healthcare", "dental": "healthcare", "clinic": "healthcare",
    "health": "healthcare", "hospital": "healthcare", "pharma": "healthcare",
    # Finance
    "finance": "finance", "banking": "finance", "insurance": "finance",
    "fintech": "finance", "accounting": "finance", "financial": "finance",
    # Legal
    "legal": "legal", "law": "legal", "attorney": "legal", "lawyer": "legal",
    # Real Estate
    "real estate": "real_estate", "property": "real_estate", "realtor": "real_estate",
    "realty": "real_estate", "homes": "real_estate", "housing": "real_estate",
    # Fitness
    "fitness": "fitness", "gym": "fitness", "yoga": "fitness", "sports": "fitness",
    "training": "fitness", "wellness": "fitness",
    # Restaurant
    "restaurant": "restaurant", "food": "restaurant", "cafe": "restaurant",
    "catering": "restaurant", "bar": "restaurant", "dining": "restaurant",
    # E-commerce
    "ecommerce": "ecommerce", "e-commerce": "ecommerce", "retail": "ecommerce",
    "shop": "ecommerce", "store": "ecommerce", "online": "ecommerce",
    # Flooring
    "flooring": "flooring", "floor": "flooring", "carpet": "flooring",
    "tile": "flooring", "hardwood": "flooring",
    # Professional Services
    "consulting": "professional_services", "agency": "professional_services",
    "services": "professional_services", "b2b": "professional_services",
}

# Goal mapping
GOAL_KEYWORDS = {
    "awareness": ["awareness", "brand", "visibility", "reach", "know"],
    "leads": ["leads", "lead gen", "prospects", "inquiries", "contacts"],
    "sales": ["sales", "revenue", "purchase", "buy", "conversion"],
    "trust": ["trust", "credibility", "authority", "reputation"],
    "launch": ["launch", "new", "introducing", "announce", "release"],
}

# Tone mapping
TONE_KEYWORDS = {
    "professional": ["professional", "corporate", "business", "formal", "executive"],
    "bold": ["bold", "energetic", "dynamic", "powerful", "intense"],
    "friendly": ["friendly", "warm", "approachable", "casual", "relatable"],
    "luxury": ["luxury", "premium", "elegant", "sophisticated", "high-end"],
    "urgent": ["urgent", "fast", "action", "limited", "now"],
}


# =============================================================================
# INTAKE QUALIFIER AGENT
# =============================================================================

class IntakeQualifierAgent:
    """
    Agent 1: Intake Qualifier for brief validation.

    Validates incoming briefs have enough information to start production.
    First gate in the 9-phase pipeline.
    """

    def __init__(
        self,
        anthropic_client=None,
        use_claude: bool = False
    ):
        self.anthropic = anthropic_client
        self.use_claude = use_claude and anthropic_client is not None

        if self.use_claude:
            logger.info("[IntakeQualifier] Initialized with Claude-enhanced extraction")
        else:
            logger.info("[IntakeQualifier] Initialized with rule-based validation")

    def _normalize_industry(self, industry_str: str) -> str:
        """Normalize industry string to canonical form."""
        if not industry_str:
            return ""

        industry_lower = industry_str.lower().strip()

        # Direct match
        if industry_lower in INDUSTRY_MAP:
            return INDUSTRY_MAP[industry_lower]

        # Partial match
        for keyword, canonical in INDUSTRY_MAP.items():
            if keyword in industry_lower:
                return canonical

        return industry_lower

    def _detect_goals(self, text: str) -> List[str]:
        """Detect campaign goals from text."""
        if not text:
            return []

        text_lower = text.lower()
        detected = []

        for goal, keywords in GOAL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(goal)

        return detected

    def _detect_tone(self, text: str) -> Optional[str]:
        """Detect tone/style from text."""
        if not text:
            return None

        text_lower = text.lower()

        for tone, keywords in TONE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return tone

        return None

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def _find_field_value(self, brief: Dict[str, Any], field_name: str) -> Optional[Any]:
        """Find field value in brief using aliases."""
        field_def = FIELD_DEFINITIONS.get(field_name)
        if not field_def:
            return None

        # Direct match
        if field_name in brief and brief[field_name]:
            return brief[field_name]

        # Check aliases
        aliases = field_def.get("aliases", [])
        for alias in aliases:
            if alias in brief and brief[alias]:
                return brief[alias]

        # Check nested structures
        for key, value in brief.items():
            if isinstance(value, dict):
                # Recursive search in nested dicts
                for alias in [field_name] + aliases:
                    if alias in value and value[alias]:
                        return value[alias]

        return None

    def _extract_from_brief(self, brief: Dict[str, Any], business_name: Optional[str], industry: Optional[str]) -> ExtractedData:
        """Extract structured data from brief."""
        extracted = ExtractedData()

        # Business name
        extracted.business_name = (
            business_name or
            self._find_field_value(brief, "business_name") or
            brief.get("business", {}).get("name")
        )

        # Industry
        raw_industry = (
            industry or
            self._find_field_value(brief, "industry") or
            brief.get("business", {}).get("industry")
        )
        extracted.industry = self._normalize_industry(raw_industry) if raw_industry else None

        # Product/Service
        extracted.product_service = (
            self._find_field_value(brief, "product_service") or
            brief.get("business", {}).get("offering")
        )

        # Target audience
        extracted.target_audience = (
            self._find_field_value(brief, "target_audience") or
            brief.get("audience", {}).get("demographic")
        )

        # Campaign goals
        goals_val = self._find_field_value(brief, "campaign_goals")
        if goals_val:
            if isinstance(goals_val, list):
                extracted.campaign_goals = goals_val
            elif isinstance(goals_val, str):
                extracted.campaign_goals = self._detect_goals(goals_val)
        else:
            # Try to detect from description
            desc = str(brief.get("description", ""))
            extracted.campaign_goals = self._detect_goals(desc)

        # Tone
        tone_val = self._find_field_value(brief, "tone")
        if tone_val:
            extracted.tone = self._detect_tone(str(tone_val)) or tone_val
        else:
            # Try to detect from style
            style = str(brief.get("style", ""))
            extracted.tone = self._detect_tone(style)

        # Call to action
        extracted.call_to_action = self._find_field_value(brief, "call_to_action")

        # Platforms
        platforms_val = self._find_field_value(brief, "platforms")
        if platforms_val:
            if isinstance(platforms_val, list):
                extracted.platforms = platforms_val
            elif isinstance(platforms_val, str):
                extracted.platforms = [p.strip() for p in platforms_val.split(",")]

        # Pain points
        pain_val = self._find_field_value(brief, "pain_points")
        if pain_val:
            if isinstance(pain_val, list):
                extracted.pain_points = pain_val
            elif isinstance(pain_val, str):
                extracted.pain_points = [pain_val]
        else:
            # Check audience.pain_points
            audience_pain = brief.get("audience", {}).get("pain_points", [])
            if audience_pain:
                extracted.pain_points = audience_pain

        # Budget
        extracted.budget_range = self._find_field_value(brief, "budget_range")

        return extracted

    def _calculate_score(self, extracted: ExtractedData) -> tuple[int, List[str]]:
        """Calculate qualification score and find missing fields."""
        score = 0
        missing = []

        # Check each field
        field_checks = {
            "business_name": extracted.business_name,
            "industry": extracted.industry,
            "product_service": extracted.product_service,
            "target_audience": extracted.target_audience,
            "campaign_goals": len(extracted.campaign_goals) > 0,
            "call_to_action": extracted.call_to_action,
            "tone": extracted.tone,
            "platforms": len(extracted.platforms) > 0,
            "pain_points": len(extracted.pain_points) > 0,
            "budget_range": extracted.budget_range,
        }

        for field_name, has_value in field_checks.items():
            field_def = FIELD_DEFINITIONS.get(field_name)
            if not field_def:
                continue

            if has_value:
                score += field_def["weight"]
            elif field_def["category"] in [FieldCategory.CRITICAL, FieldCategory.IMPORTANT]:
                missing.append(field_def["description"])

        return min(100, score), missing

    def _determine_status(self, score: int, strict: bool) -> QualificationStatus:
        """Determine qualification status from score."""
        if score >= 71:
            return QualificationStatus.QUALIFIED
        elif score >= 41 and not strict:
            return QualificationStatus.NEEDS_INFO
        else:
            return QualificationStatus.REJECTED

    def _generate_suggestions(self, missing: List[str], extracted: ExtractedData) -> List[str]:
        """Generate actionable suggestions for improving brief."""
        suggestions = []

        if not extracted.business_name:
            suggestions.append("Please provide your business/company name")

        if not extracted.product_service:
            suggestions.append("What product or service will this commercial promote?")

        if not extracted.target_audience:
            suggestions.append("Who is your ideal customer/target audience?")

        if not extracted.campaign_goals:
            suggestions.append("What's the primary goal? (Awareness, Leads, Sales, Trust, Launch)")

        if not extracted.call_to_action:
            suggestions.append("What should viewers do after watching? (Visit website, Book demo, etc.)")

        if not extracted.tone:
            suggestions.append("What tone/style should the video have? (Professional, Bold, Friendly, etc.)")

        return suggestions[:3]  # Return top 3 suggestions

    async def qualify(self, request: QualificationRequest) -> QualificationResult:
        """
        Qualify a brief for video production.

        Args:
            request: QualificationRequest with brief data

        Returns:
            QualificationResult with status, score, and suggestions
        """
        logger.info("Qualifying brief...")

        # Extract structured data
        extracted = self._extract_from_brief(
            brief=request.brief,
            business_name=request.business_name,
            industry=request.industry
        )

        # Calculate score and find missing
        score, missing = self._calculate_score(extracted)

        # Determine status
        status = self._determine_status(score, request.strict)

        # Generate suggestions
        suggestions = self._generate_suggestions(missing, extracted)

        # Generate warnings
        warnings = []
        if status == QualificationStatus.NEEDS_INFO:
            warnings.append(f"Brief qualified with warnings. Score: {score}/100")
            warnings.append(f"Missing: {', '.join(missing[:3])}")

        logger.info(f"Qualification complete: {status.value} ({score}/100)")

        return QualificationResult(
            status=status,
            score=score,
            missing_fields=missing,
            suggestions=suggestions,
            extracted_data=extracted,
            warnings=warnings,
            source="rule_based"
        )

    async def qualify_dict(
        self,
        brief: Dict[str, Any],
        business_name: Optional[str] = None,
        industry: Optional[str] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Convenience method for dict input/output."""
        request = QualificationRequest(
            brief=brief,
            business_name=business_name,
            industry=industry,
            strict=strict
        )
        result = await self.qualify(request)
        return {
            "status": result.status.value,
            "score": result.score,
            "missing_fields": result.missing_fields,
            "suggestions": result.suggestions,
            "extracted_data": result.extracted_data.model_dump(),
            "warnings": result.warnings,
            "source": result.source
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_intake_qualifier(
    anthropic_client=None,
    use_claude: bool = False
) -> IntakeQualifierAgent:
    """Create IntakeQualifierAgent instance."""
    return IntakeQualifierAgent(
        anthropic_client=anthropic_client,
        use_claude=use_claude
    )


# =============================================================================
# SIMPLE VALIDATION HELPER
# =============================================================================

def basic_brief_validation(
    brief: Dict[str, Any],
    business_name: Optional[str] = None,
    industry: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simple synchronous validation without agent.
    Useful for quick checks before full qualification.
    """
    score = 100
    missing = []

    # Check critical fields
    if not (business_name or brief.get("business_name") or brief.get("business", {}).get("name")):
        missing.append("business_name")
        score -= 20

    if not (industry or brief.get("industry") or brief.get("business", {}).get("industry")):
        missing.append("industry")
        score -= 15

    # Check for product/service description
    has_product = any([
        brief.get("description"),
        brief.get("product"),
        brief.get("service"),
        brief.get("offering"),
        brief.get("business", {}).get("offering")
    ])
    if not has_product:
        missing.append("product/service description")
        score -= 20

    # Check for target audience
    has_audience = any([
        brief.get("target_audience"),
        brief.get("audience"),
        brief.get("demographic"),
        brief.get("audience", {}).get("demographic") if isinstance(brief.get("audience"), dict) else None
    ])
    if not has_audience:
        missing.append("target_audience")
        score -= 15

    # Check for goals
    has_goals = any([
        brief.get("goals"),
        brief.get("campaign_goals"),
        brief.get("objectives"),
        brief.get("video_goal")
    ])
    if not has_goals:
        missing.append("campaign_goals")
        score -= 10

    # Determine status
    score = max(0, score)
    if score >= 71:
        status = "qualified"
    elif score >= 41:
        status = "needs_info"
    else:
        status = "rejected"

    return {
        "status": status,
        "score": score,
        "missing_fields": missing,
        "suggestions": [f"Please provide: {', '.join(missing)}"] if missing else []
    }


# =============================================================================
# TESTING
# =============================================================================

async def test_intake_qualifier():
    """Test the intake qualifier agent."""
    agent = create_intake_qualifier()

    print("\n[IntakeQualifier] Test Mode")
    print("=" * 60)

    # Test 1: Complete brief
    complete_brief = {
        "business_name": "TechCorp Solutions",
        "industry": "technology",
        "product": "AI-powered analytics platform",
        "target_audience": "Marketing directors at B2B SaaS companies",
        "goals": ["lead generation", "awareness"],
        "tone": "professional",
        "call_to_action": "Book a demo",
        "platforms": ["LinkedIn", "YouTube"]
    }

    print("\nTest 1: Complete brief")
    result1 = await agent.qualify_dict(complete_brief)
    print(f"  Status: {result1['status']}")
    print(f"  Score: {result1['score']}/100")
    print(f"  Missing: {result1['missing_fields']}")

    # Test 2: Minimal brief
    minimal_brief = {
        "business": {
            "name": "Joe's Pizza",
            "industry": "restaurant"
        },
        "description": "Best pizza in town"
    }

    print("\nTest 2: Minimal brief")
    result2 = await agent.qualify_dict(minimal_brief)
    print(f"  Status: {result2['status']}")
    print(f"  Score: {result2['score']}/100")
    print(f"  Missing: {result2['missing_fields']}")
    print(f"  Suggestions: {result2['suggestions']}")

    # Test 3: Empty brief
    empty_brief = {}

    print("\nTest 3: Empty brief")
    result3 = await agent.qualify_dict(empty_brief)
    print(f"  Status: {result3['status']}")
    print(f"  Score: {result3['score']}/100")
    print(f"  Missing: {result3['missing_fields']}")


if __name__ == "__main__":
    asyncio.run(test_intake_qualifier())
