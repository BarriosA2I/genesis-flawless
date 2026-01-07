"""
================================================================================
KNOWLEDGE LOADER - AI Creative Director Memory Integration
================================================================================
Loads and manages the Memory Pack v2.0 knowledge base for the Creative Director.

Author: Barrios A2I | December 2025
================================================================================
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger("KnowledgeLoader")

# Knowledge base path
KNOWLEDGE_BASE = Path(__file__).parent / "knowledge" / "Barrios_A2I_Creative_Director_Memory_Pack"


class KnowledgeLoader:
    """
    Loads and provides access to the Creative Director's knowledge base.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or KNOWLEDGE_BASE
        self._cache: Dict[str, str] = {}
        self._loaded = False

    def load_all(self) -> Dict[str, str]:
        """Load all knowledge files into memory."""
        if self._loaded:
            return self._cache

        if not self.base_path.exists():
            logger.warning(f"Knowledge base not found at {self.base_path}")
            return {}

        # Load all markdown files
        for md_file in self.base_path.rglob("*.md"):
            try:
                relative_path = md_file.relative_to(self.base_path)
                with open(md_file, 'r', encoding='utf-8') as f:
                    self._cache[str(relative_path)] = f.read()
            except Exception as e:
                logger.error(f"Error loading {md_file}: {e}")

        # Load all JSON files
        for json_file in self.base_path.rglob("*.json"):
            try:
                relative_path = json_file.relative_to(self.base_path)
                with open(json_file, 'r', encoding='utf-8') as f:
                    self._cache[str(relative_path)] = f.read()
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._cache)} knowledge files")
        return self._cache

    def get_file(self, relative_path: str) -> Optional[str]:
        """Get a specific knowledge file by relative path."""
        if not self._loaded:
            self.load_all()
        return self._cache.get(relative_path)

    def _normalize_path(self, file_path: str) -> Optional[str]:
        """Find the cached key for a file path, handling OS differences."""
        # Try exact match first
        if file_path in self._cache:
            return file_path

        # Try with backslashes (Windows)
        backslash_path = file_path.replace("/", "\\")
        if backslash_path in self._cache:
            return backslash_path

        # Try with forward slashes (Unix)
        forward_path = file_path.replace("\\", "/")
        if forward_path in self._cache:
            return forward_path

        return None

    def get_business_knowledge(self) -> Dict[str, str]:
        """Get all business-related knowledge for sales/discovery."""
        if not self._loaded:
            self.load_all()

        business_docs = {}
        business_files = [
            "06_business/company_overview.md",
            "06_business/services_catalog.md",
            "06_business/faqs.md",
            "06_business/objection_handling.md",
            "06_business/differentiators.md",
            "06_business/technology_overview.md",
            "06_business/founder_profile.md",
            "06_business/target_customers.md",
            "06_business/conversation_examples.md"
        ]

        for file_path in business_files:
            normalized = self._normalize_path(file_path)
            if normalized and normalized in self._cache:
                # Use filename without extension as key
                key = Path(file_path).stem
                business_docs[key] = self._cache[normalized]

        return business_docs

    def get_brand_knowledge(self) -> Dict[str, str]:
        """Get brand/visual knowledge."""
        if not self._loaded:
            self.load_all()

        brand_docs = {}
        brand_files = [
            "01_brand/brand_identity.md",
            "01_brand/design_tokens.json",
            "01_brand/typography.md"
        ]

        for file_path in brand_files:
            normalized = self._normalize_path(file_path)
            if normalized and normalized in self._cache:
                key = Path(file_path).stem
                brand_docs[key] = self._cache[normalized]

        return brand_docs

    def get_product_knowledge(self) -> Dict[str, str]:
        """Get product/intake knowledge."""
        if not self._loaded:
            self.load_all()

        product_docs = {}
        product_files = [
            "02_product/creative_director_product_spec.md",
            "02_product/intake_schema.json"
        ]

        for file_path in product_files:
            normalized = self._normalize_path(file_path)
            if normalized and normalized in self._cache:
                key = Path(file_path).stem
                product_docs[key] = self._cache[normalized]

        return product_docs

    def get_prompting_knowledge(self) -> Dict[str, str]:
        """Get prompting/production knowledge."""
        if not self._loaded:
            self.load_all()

        prompting_docs = {}
        prompting_files = [
            "03_prompting/guardrails.md",
            "03_prompting/veo_3_scene_template.md",
            "03_prompting/sora_2_script_template.md",
            "03_prompting/nanobanana_image_json_template.json"
        ]

        for file_path in prompting_files:
            normalized = self._normalize_path(file_path)
            if normalized and normalized in self._cache:
                key = Path(file_path).stem
                prompting_docs[key] = self._cache[normalized]

        return prompting_docs

    def get_system_instructions(self) -> Dict[str, str]:
        """Get master system instructions."""
        if not self._loaded:
            self.load_all()

        system_docs = {}
        system_files = [
            "05_system/master_instructions.md",
            "05_system/agent_roles.md"
        ]

        for file_path in system_files:
            normalized = self._normalize_path(file_path)
            if normalized and normalized in self._cache:
                key = Path(file_path).stem
                system_docs[key] = self._cache[normalized]

        return system_docs

    def get_examples(self) -> Dict[str, str]:
        """Get example briefs and prompts."""
        if not self._loaded:
            self.load_all()

        example_docs = {}
        example_files = [
            "04_examples/brief_template.md",
            "04_examples/example_veo_prompt_32s.md"
        ]

        for file_path in example_files:
            normalized = self._normalize_path(file_path)
            if normalized and normalized in self._cache:
                key = Path(file_path).stem
                example_docs[key] = self._cache[normalized]

        return example_docs

    def get_context_for_mode(self, mode: str) -> str:
        """
        Get relevant knowledge context based on conversation mode.

        Args:
            mode: 'discovery', 'intake', or 'production'

        Returns:
            Formatted knowledge context string
        """
        context_parts = []

        if mode == "discovery":
            # Include business knowledge for answering questions
            business = self.get_business_knowledge()
            context_parts.append("## BUSINESS KNOWLEDGE\n")
            for key, content in business.items():
                context_parts.append(f"### {key.replace('_', ' ').title()}\n{content}\n\n")

        elif mode == "intake":
            # Include product knowledge for brief intake
            product = self.get_product_knowledge()
            context_parts.append("## INTAKE PROCESS\n")
            for key, content in product.items():
                context_parts.append(f"### {key.replace('_', ' ').title()}\n{content}\n\n")

        elif mode == "production":
            # Include prompting knowledge for video production
            prompting = self.get_prompting_knowledge()
            context_parts.append("## PRODUCTION TEMPLATES\n")
            for key, content in prompting.items():
                context_parts.append(f"### {key.replace('_', ' ').title()}\n{content}\n\n")

            # Also include examples
            examples = self.get_examples()
            context_parts.append("## EXAMPLES\n")
            for key, content in examples.items():
                context_parts.append(f"### {key.replace('_', ' ').title()}\n{content}\n\n")

        return "\n".join(context_parts)

    def build_full_system_prompt(self) -> str:
        """
        Build the complete system prompt with all knowledge integrated.

        Returns:
            Full system prompt string
        """
        # Get master instructions
        system_docs = self.get_system_instructions()
        master_instructions = system_docs.get("master_instructions", "")

        # Get condensed business context (key points only)
        business = self.get_business_knowledge()
        services = business.get("services_catalog", "")
        faqs = business.get("faqs", "")
        objections = business.get("objection_handling", "")

        # Build comprehensive prompt
        prompt = f"""# AI CREATIVE DIRECTOR - BARRIOS A2I
{master_instructions}

## KEY BUSINESS FACTS (Reference when answering questions)

### Services & Pricing
{services[:3000]}

### FAQ Reference
{faqs[:2000]}

### Objection Handling
{objections[:3000]}
"""
        return prompt


def route_to_knowledge(user_message: str) -> str:
    """
    Determine which conversation mode based on user intent.

    Args:
        user_message: The user's message

    Returns:
        Mode string: 'discovery', 'intake', or 'production'
    """
    message_lower = user_message.lower()

    # Intake triggers - user wants to start creating
    intake_triggers = [
        "start", "begin", "let's do", "let's go", "ready",
        "create commercial", "make video", "new project",
        "start my brief", "start the brief", "create my",
        "i want to", "i need a commercial", "i'd like to"
    ]

    # Production triggers - user is in production phase
    production_triggers = [
        "generate prompt", "veo prompt", "sora prompt", "runway",
        "video prompt", "scene prompt", "script prompt",
        "create the video", "produce", "production"
    ]

    # Business/Sales triggers - user asking questions
    business_triggers = [
        "what is", "who is", "how much", "price", "cost",
        "why should", "different", "competitor", "trust",
        "work with", "services", "offer", "gary", "barrios",
        "how does", "what do you", "tell me about",
        "explain", "how are you", "what makes", "expensive",
        "cheap", "guarantee", "turnaround", "delivery"
    ]

    # Check triggers in priority order
    if any(trigger in message_lower for trigger in intake_triggers):
        return "intake"
    elif any(trigger in message_lower for trigger in production_triggers):
        return "production"
    elif any(trigger in message_lower for trigger in business_triggers):
        return "discovery"
    else:
        return "discovery"  # Default to discovery mode


# Global knowledge loader instance
_knowledge_loader: Optional[KnowledgeLoader] = None


def get_knowledge_loader() -> KnowledgeLoader:
    """Get or create the global knowledge loader instance."""
    global _knowledge_loader
    if _knowledge_loader is None:
        _knowledge_loader = KnowledgeLoader()
        _knowledge_loader.load_all()
    return _knowledge_loader


def get_knowledge_context(mode: str) -> str:
    """
    Convenience function to get knowledge context for a mode.

    Args:
        mode: 'discovery', 'intake', or 'production'

    Returns:
        Knowledge context string
    """
    loader = get_knowledge_loader()
    return loader.get_context_for_mode(mode)
