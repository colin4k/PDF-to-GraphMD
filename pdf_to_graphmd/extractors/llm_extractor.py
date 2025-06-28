"""
LLM-based knowledge extraction module
"""
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from dataclasses import asdict

from ..models import Entity, Relation, KnowledgeTriple, KnowledgeGraph, DocumentContent
from ..config import LLMConfig, OntologyConfig, OutputConfig


logger = logging.getLogger(__name__)


class LLMExtractor:
    """LLM-based knowledge extraction using structured prompts"""
    
    def __init__(self, llm_config: LLMConfig, ontology_config: OntologyConfig, output_config: OutputConfig = None):
        self.llm_config = llm_config
        self.ontology_config = ontology_config
        self.output_config = output_config
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup LLM client (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            
            self.client = OpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url
            )
            logger.info(f"LLM client initialized with model: {self.llm_config.model_name}")
            
        except ImportError:
            logger.error("OpenAI library not found. Please install: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
    
    def extract_knowledge(self, content: DocumentContent) -> KnowledgeGraph:
        """
        Extract knowledge graph from document content using LLM
        
        Args:
            content: Parsed document content
            
        Returns:
            KnowledgeGraph with extracted entities and relations
        """
        try:
            logger.info("Starting LLM-based knowledge extraction")
            
            # Split content into chunks for processing
            text_chunks = self._split_text(content.text)
            
            # Extract entities and relations from each chunk
            all_entities = {}
            all_relations = []
            all_triples = []
            
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                
                entities, relations = self._extract_from_chunk(chunk)
                
                # Merge entities (avoid duplicates)
                for entity in entities:
                    if entity.id not in all_entities:
                        all_entities[entity.id] = entity
                    else:
                        # Merge information from duplicate entities
                        existing = all_entities[entity.id]
                        existing.description = existing.description or entity.description
                        existing.aliases.extend([a for a in entity.aliases if a not in existing.aliases])
                        existing.attributes.update(entity.attributes)
                
                all_relations.extend(relations)
                
                # Create triples
                for relation in relations:
                    if relation.source_entity in all_entities and relation.target_entity in all_entities:
                        triple = KnowledgeTriple(
                            subject=all_entities[relation.source_entity],
                            predicate=relation.relation_type,
                            object=all_entities[relation.target_entity],
                            confidence=relation.confidence,
                            source_text=chunk,
                            source_document=""  # Will be set by caller
                        )
                        all_triples.append(triple)
            
            # Create knowledge graph
            kg = KnowledgeGraph(
                entities=all_entities,
                relations=all_relations,
                triples=all_triples
            )
            
            logger.info(f"Extracted {len(all_entities)} entities and {len(all_relations)} relations")
            return kg
            
        except Exception as e:
            logger.error(f"Error in LLM knowledge extraction: {str(e)}")
            raise
    
    def _split_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split text into manageable chunks for LLM processing"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_from_chunk(self, text_chunk: str) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from a text chunk using LLM"""
        try:
            logger.info(f"Extracting from chunk (length: {len(text_chunk)} chars)")
            logger.debug(f"Chunk preview: {text_chunk[:200]}...")
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(text_chunk)
            
            # Call LLM
            logger.info("Calling LLM API...")
            response = self._call_llm(prompt)
            logger.info(f"LLM response received (length: {len(response)} chars)")
            logger.debug(f"LLM response preview: {response[:500]}...")
            
            # Parse response
            entities, relations = self._parse_llm_response(response)
            logger.info(f"Parsed {len(entities)} entities and {len(relations)} relations from chunk")
            
            return entities, relations
            
        except Exception as e:
            logger.error(f"Error extracting from chunk: {str(e)}")
            return [], []
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create structured prompt for knowledge extraction"""
        
        entity_types_str = ", ".join(self.ontology_config.entity_types)
        relation_types_str = ", ".join(self.ontology_config.relation_types)
        
        # Get language-specific prompts if available
        if self.output_config and hasattr(self.output_config, 'language') and hasattr(self.output_config, 'language_prompts'):
            language = getattr(self.output_config, 'language', 'en')
            language_prompts = getattr(self.output_config, 'language_prompts', {})
            
            if language in language_prompts:
                lang_config = language_prompts[language]
                base_system_prompt = lang_config.get('system_prompt', "You are an expert knowledge extraction system.")
                entity_prompt = lang_config.get('entity_prompt', "Extract entities from the following text:")
                relation_prompt = lang_config.get('relation_prompt', "Analyze relationships between entities:")
            else:
                # Fallback to English
                base_system_prompt = "You are an expert knowledge extraction system."
                entity_prompt = "Extract entities from the following text:"
                relation_prompt = "Analyze relationships between entities:"
        else:
            # Default English prompts
            base_system_prompt = "You are an expert knowledge extraction system."
            entity_prompt = "Extract entities from the following text:"
            relation_prompt = "Analyze relationships between entities:"
        
        if self.output_config and getattr(self.output_config, 'language', 'en') == 'zh':
            # Chinese prompts
            system_prompt = f"""{base_system_prompt}你的任务是从给定文本中提取实体及其关系。

实体类型（仅使用这些类型）: {entity_types_str}
关系类型（仅使用这些类型）: {relation_types_str}

按以下JSON格式提取信息：
{{
  "entities": [
    {{
      "id": "唯一标识符",
      "name": "实体名称",
      "type": "实体类型",
      "description": "简要描述",
      "aliases": ["别名1", "别名2"]
    }}
  ],
  "relations": [
    {{
      "source_entity": "源实体id",
      "relation_type": "关系类型",
      "target_entity": "目标实体id",
      "description": "关系描述",
      "confidence": 0.95
    }}
  ]
}}

规则：
1. 仅提取有意义的实体和关系
2. 使用描述性但简洁的实体ID
3. 确保关系中的所有实体ID都存在于实体列表中
4. 根据清晰度分配置信度分数（0.0-1.0）
5. 如果提到替代名称，请在别名中包含"""

            user_prompt = f"从以下文本中提取知识：\n\n{text}"
        else:
            # English prompts
            system_prompt = f"""{base_system_prompt} Your task is to extract entities and their relationships from the given text.

ENTITY TYPES (use only these): {entity_types_str}
RELATION TYPES (use only these): {relation_types_str}

Extract information in the following JSON format:
{{
  "entities": [
    {{
      "id": "unique_identifier",
      "name": "entity_name",
      "type": "entity_type",
      "description": "brief_description",
      "aliases": ["alternative_name1", "alternative_name2"]
    }}
  ],
  "relations": [
    {{
      "source_entity": "source_entity_id",
      "relation_type": "relation_type",
      "target_entity": "target_entity_id",
      "description": "relation_description",
      "confidence": 0.95
    }}
  ]
}}

Rules:
1. Extract only meaningful entities and relations
2. Use descriptive but concise entity IDs
3. Ensure all entity IDs in relations exist in the entities list
4. Assign confidence scores (0.0-1.0) based on clarity
5. Include alternative names in aliases if mentioned"""

            user_prompt = f"Extract knowledge from this text:\n\n{text}"
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM API with structured prompts"""
        try:
            logger.info(f"Calling LLM with model: {self.llm_config.model_name}")
            logger.debug(f"Messages: {messages}")
            
            if self.llm_config.force_json:
                # Use JSON mode if available
                logger.info("Using JSON response format")
                response = self.client.chat.completions.create(
                    model=self.llm_config.model_name,
                    messages=messages,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    response_format={"type": "json_object"}
                )
            else:
                logger.info("Using standard response format")
                response = self.client.chat.completions.create(
                    model=self.llm_config.model_name,
                    messages=messages,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens
                )
            
            content = response.choices[0].message.content
            logger.info(f"LLM API call successful, response length: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                logger.error(f"API response: {e.response}")
            raise
    
    def _parse_llm_response(self, response: str) -> Tuple[List[Entity], List[Relation]]:
        """Parse LLM response and create Entity/Relation objects"""
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {repr(response)}")
            
            # Clean response (remove markdown formatting if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Check if response is empty or whitespace only
            if not cleaned_response or cleaned_response.isspace():
                logger.warning("LLM response is empty or whitespace only")
                return [], []
            
            # Try to parse JSON
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Cleaned response: {repr(cleaned_response)}")
                
                # Try to extract JSON from the response if it's wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        logger.info("Successfully extracted JSON from wrapped response")
                    except json.JSONDecodeError:
                        logger.error("Failed to extract valid JSON from response")
                        return [], []
                else:
                    logger.error("No JSON object found in response")
                    return [], []
            
            entities = []
            relations = []
            
            # Create Entity objects
            for entity_data in data.get("entities", []):
                entity = Entity(
                    id=entity_data.get("id", ""),
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", ""),
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    confidence=entity_data.get("confidence", 1.0)
                )
                entities.append(entity)
            
            # Create Relation objects
            for relation_data in data.get("relations", []):
                relation = Relation(
                    source_entity=relation_data.get("source_entity", ""),
                    relation_type=relation_data.get("relation_type", ""),
                    target_entity=relation_data.get("target_entity", ""),
                    description=relation_data.get("description", ""),
                    confidence=relation_data.get("confidence", 1.0)
                )
                relations.append(relation)
            
            return entities, relations
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.error(f"Response content: {repr(response)}")
            return [], []
    
    def validate_extraction(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Validate extracted knowledge graph"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {
                "entities": len(kg.entities),
                "relations": len(kg.relations),
                "triples": len(kg.triples)
            }
        }
        
        # Check for orphaned relations
        entity_ids = set(kg.entities.keys())
        for relation in kg.relations:
            if relation.source_entity not in entity_ids:
                validation_results["warnings"].append(
                    f"Relation source entity not found: {relation.source_entity}"
                )
            if relation.target_entity not in entity_ids:
                validation_results["warnings"].append(
                    f"Relation target entity not found: {relation.target_entity}"
                )
        
        # Check entity types
        valid_types = set(self.ontology_config.entity_types)
        for entity in kg.entities.values():
            if entity.type not in valid_types:
                validation_results["warnings"].append(
                    f"Invalid entity type '{entity.type}' for entity '{entity.name}'"
                )
        
        # Check relation types
        valid_relations = set(self.ontology_config.relation_types)
        for relation in kg.relations:
            if relation.relation_type not in valid_relations:
                validation_results["warnings"].append(
                    f"Invalid relation type: {relation.relation_type}"
                )
        
        if validation_results["warnings"] or validation_results["errors"]:
            validation_results["valid"] = False
        
        return validation_results