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
        """Extract entities and relations from a text chunk using LLM with retry mechanism"""
        logger.info(f"Extracting from chunk (length: {len(text_chunk)} chars)")
        logger.debug(f"Chunk preview: {text_chunk[:200]}...")
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(text_chunk)
        
        max_retries = self.llm_config.max_retries
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Calling LLM API...")
                response = self._call_llm(prompt)
                logger.info(f"LLM response received (length: {len(response)} chars)")
                logger.debug(f"LLM response preview: {response[:500]}...")
                
                # Parse response
                entities, relations = self._parse_llm_response(response)
                logger.info(f"Parsed {len(entities)} entities and {len(relations)} relations from chunk")
                
                return entities, relations
                
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with JSON decode error: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed due to JSON decode errors")
                    return [], []
                else:
                    logger.info(f"Retrying... ({max_retries - attempt - 1} attempts remaining)")
                    
            except Exception as e:
                logger.error(f"Error extracting from chunk on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
                    return [], []
                else:
                    logger.info(f"Retrying... ({max_retries - attempt - 1} attempts remaining)")
        
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
            system_prompt = f"""{base_system_prompt}你是一个专业的知识图谱分析系统。你的任务是分析文本内容，提取知识点并建立它们之间的连接。

**核心任务：**
利用知识图谱来分析文本，以JSON格式输出每个知识点的定义及描述，并在描述中使用Obsidian支持的"[[]]"语法来建立知识点之间的连接。

**关键要求 - 必须严格遵守：**
- **所有输出必须使用标准简体中文汉字**
- **如果文本是繁体中文，需要转换为符合简体中文语境的简体中文文本**
- **绝对不允许出现任何繁体字**
- **示例对比：**
  - ✅ 正确：经济、设备、时间、发展、应该、导致、阶段
  - ❌ 错误：經濟、設備、時間、發展、應該、導致、階段
- **在实体描述中主动使用[[]]语法链接相关知识点**
- **不要包含如[cite:start]或[cite:1]等非标准语法的内容**

实体类型（仅使用这些类型）: {entity_types_str}
关系类型（仅使用这些类型）: {relation_types_str}

按以下JSON格式提取信息：
{{
  "entities": [
    {{
      "id": "唯一标识符",
      "name": "实体名称",
      "type": "实体类型",
      "description": "详细描述（在描述中使用[[]]语法链接到其他相关实体）",
      "aliases": ["别名1", "别名2"]
    }}
  ],
  "relations": [
    {{
      "source_entity": "源实体id",
      "relation_type": "关系类型",
      "target_entity": "目标实体id",
      "description": "关系描述（可在描述中使用[[]]语法）",
      "confidence": 0.95
    }}
  ]
}}

规则：
1. 提取所有重要知识点作为实体，构建丰富的知识网络
2. **在实体描述中主动使用[[实体名称]]格式链接到其他相关实体**
3. 使用描述性但简洁的实体ID
4. 确保关系中的所有实体ID都存在于实体列表中
5. 根据清晰度分配置信度分数（0.0-1.0）
6. 如果提到替代名称，请在别名中包含
7. **🚨 重要：所有中文内容必须使用简体中文，绝对禁止繁体字**
8. **🚨 重要：在描述中积极建立[[]]链接，体现知识点之间的关系**
9. **积极寻找实体间的各种关系，构建连通的知识图谱**
10. **输出前检查：确保没有出现經濟、設備、時間、應該、導致等繁体字**
11. **输出前检查：确保没有[cite:start]等非标准语法**"""

            user_prompt = f"""利用知识图谱来分析以下文本，提取每个知识点的定义及描述，并在描述中使用Obsidian支持的"[[]]"语法来建立知识点之间的连接。

**特别要求：**
1. 提取所有重要概念、人物、组织、地点等作为实体
2. 在实体描述中主动使用[[]]语法链接相关知识点
3. 如果文本是繁体中文，需要转换为符合简体中文语境的简体中文文本
4. **🚨 关键：必须用标准简体中文输出所有内容**
5. **🚨 关键：禁止使用任何繁体字（如經濟、設備、時間等）**
6. **🚨 关键：在描述中积极建立[[]]双向链接**
7. **不要包含如[cite:start]或[cite:1]等非markdown语法的内容**

文本内容：
{text}"""
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
            raise json.JSONDecodeError("Empty or whitespace-only response", cleaned_response, 0)
        
        # Try to parse JSON - let JSONDecodeError propagate for retry mechanism
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
                    # Re-raise the original error for retry mechanism
                    raise e
            else:
                logger.error("No JSON object found in response")
                # Re-raise the original error for retry mechanism
                raise e
        
        try:
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
            logger.error(f"Error creating entities/relations from parsed JSON: {str(e)}")
            logger.error(f"Parsed data: {repr(data)}")
            raise
    
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