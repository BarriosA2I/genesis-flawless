"""
NEXUS Brain - Knowledge System for Barrios A2I Commercial Lab
Loads and formats knowledge from YAML files for AI system prompt injection.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not installed. NEXUS Brain unavailable.")

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Knowledge directory (relative to project root)
KNOWLEDGE_DIR = Path(__file__).parent.parent / "config" / "knowledge"

# ============================================================================
# NEXUS BRAIN CLASS
# ============================================================================

class NexusBrain:
    """Loads and manages knowledge from YAML files for AI context injection."""

    def __init__(self, knowledge_dir: Path = None):
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        self.pricing: Dict[str, Any] = {}
        self.services: Dict[str, Any] = {}
        self.faqs: Dict[str, Any] = {}
        self._loaded = False

        self._load_all()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file from the knowledge directory."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available")
            return {}

        filepath = self.knowledge_dir / filename
        if not filepath.exists():
            logger.warning(f"Knowledge file not found: {filepath}")
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                logger.info(f"Loaded knowledge: {filename}")
                return data or {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}

    def _load_all(self):
        """Load all knowledge files."""
        self.pricing = self._load_yaml("pricing.yaml")
        self.services = self._load_yaml("services.yaml")
        self.faqs = self._load_yaml("faqs.yaml")
        self._loaded = True

    def reload(self):
        """Reload all knowledge files (hot reload)."""
        self._load_all()
        logger.info("NEXUS Brain knowledge reloaded")

    # ========================================================================
    # CONTEXT FORMATTERS
    # ========================================================================

    def get_pricing_context(self) -> str:
        """Format pricing knowledge for system prompt."""
        if not self.pricing:
            return "Pricing information not available."

        lines = ["## PRICING"]

        # Subscription tiers
        tiers = self.pricing.get("tiers", {})
        for tier_key, tier in tiers.items():
            name = tier.get("name", tier_key.title())
            price = tier.get("price", "?")
            tokens = tier.get("tokens", "?")
            queue = tier.get("queue", "?")
            lines.append(f"- {name}: ${price}/mo, {tokens} tokens, {queue} queue")

        # Lab test
        lab = self.pricing.get("lab_test", {})
        if lab:
            lines.append(f"- Lab Test: ${lab.get('price', 500)} one-time, 8 tokens")

        # Token math
        token_math = self.pricing.get("token_math", {})
        if token_math:
            lines.append(f"\nToken math: {token_math.get('explanation', '')}")
            lines.append(f"Rollover: {token_math.get('rollover_note', 'No rollover')}")

        return "\n".join(lines)

    def get_services_context(self) -> str:
        """Format services knowledge for system prompt."""
        if not self.services:
            return "Services information not available."

        lines = ["## SERVICES"]

        lab = self.services.get("commercial_lab", {})
        if lab:
            lines.append(f"Commercial Lab: {lab.get('description', '')}")

        # Deliverables
        deliverables = lab.get("deliverables", [])
        if deliverables:
            lines.append("Deliverables: " + ", ".join([d.get("name", "") for d in deliverables]))

        # Platforms
        platforms = lab.get("platforms", {}).get("supported", [])
        if platforms:
            lines.append(f"Platforms: {', '.join(platforms)}")

        return "\n".join(lines)

    def get_faqs_context(self) -> str:
        """Format FAQs for system prompt."""
        if not self.faqs:
            return "FAQs not available."

        lines = ["## QUICK ANSWERS"]

        faqs = self.faqs.get("faqs", [])
        for faq in faqs[:8]:  # Limit to 8 most important
            q = faq.get("question", "")
            a = faq.get("answer", "")
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")

        return "\n".join(lines)

    def get_full_context(self) -> str:
        """Get all knowledge formatted for system prompt injection."""
        sections = [
            self.get_pricing_context(),
            self.get_services_context(),
            self.get_faqs_context()
        ]
        return "\n\n".join(sections)

    # ========================================================================
    # QUERY HELPERS
    # ========================================================================

    def get_tier_info(self, tier_name: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific pricing tier."""
        return self.pricing.get("tiers", {}).get(tier_name.lower())

    def get_addon_info(self, addon_name: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific add-on."""
        return self.pricing.get("add_ons", {}).get(addon_name.lower())

    def find_faq(self, query: str) -> Optional[Dict[str, Any]]:
        """Find an FAQ that matches the query."""
        query_lower = query.lower()
        faqs = self.faqs.get("faqs", [])

        for faq in faqs:
            keywords = faq.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return faq

        return None

    def format_tier_response(self, tier_name: str) -> str:
        """Format a friendly response about a pricing tier."""
        tier = self.get_tier_info(tier_name)
        if not tier:
            return f"I don't have info on the {tier_name} tier."

        return (
            f"{tier.get('name', tier_name.title())} is ${tier.get('price', '?')}/month. "
            f"You get {tier.get('tokens', '?')} tokens ({tier.get('commercials', '?')} commercial{'s' if tier.get('commercials', 1) > 1 else ''}), "
            f"{tier.get('queue', '?')} turnaround."
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_brain: Optional[NexusBrain] = None

def get_brain() -> NexusBrain:
    """Get or create the global NEXUS Brain instance."""
    global _brain
    if _brain is None:
        _brain = NexusBrain()
    return _brain


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_knowledge_context() -> str:
    """Get full knowledge context for system prompt."""
    return get_brain().get_full_context()

def get_pricing_context() -> str:
    """Get pricing context for system prompt."""
    return get_brain().get_pricing_context()

def find_faq_answer(query: str) -> Optional[str]:
    """Find an FAQ answer for a query."""
    faq = get_brain().find_faq(query)
    return faq.get("answer") if faq else None
