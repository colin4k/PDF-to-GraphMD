"""
Obsidian-compatible Markdown file generation module
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import yaml
import re
from datetime import datetime

from ..models import KnowledgeGraph, Entity, Relation, ObsidianNote, DocumentContent
from ..config import OutputConfig


logger = logging.getLogger(__name__)


class ObsidianGenerator:
    """Generate Obsidian-compatible vault from knowledge graph"""
    
    def __init__(self, output_config: OutputConfig):
        self.output_config = output_config
        self.vault_path = Path(output_config.output_dir)
        self.assets_path = self.vault_path / "assets"
        
    def generate_vault(self, knowledge_graph: KnowledgeGraph, 
                      document_content: Optional[DocumentContent] = None,
                      source_file: str = "") -> List[ObsidianNote]:
        """
        Generate complete Obsidian vault from knowledge graph
        
        Args:
            knowledge_graph: The knowledge graph to convert
            document_content: Original document content for embedding assets
            source_file: Source PDF file path
            
        Returns:
            List of generated ObsidianNote objects
        """
        try:
            logger.info(f"Generating Obsidian vault at: {self.vault_path}")
            
            # Ensure output directory exists
            self.vault_path.mkdir(parents=True, exist_ok=True)
            
            # Generate notes for each entity
            notes = []
            for entity_id, entity in knowledge_graph.entities.items():
                note = self._generate_entity_note(entity, knowledge_graph, document_content)
                notes.append(note)
            
            # Save all notes
            for note in notes:
                note.save(self.vault_path)
            
            # Generate index note
            index_note = self._generate_index_note(knowledge_graph, source_file)
            index_note.save(self.vault_path)
            notes.append(index_note)
            
            # Copy assets if available
            if document_content and self.output_config.include_images:
                self._copy_assets(document_content)
            
            # Generate graph view configuration
            self._generate_graph_config()
            
            logger.info(f"Generated {len(notes)} notes in Obsidian vault")
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating Obsidian vault: {str(e)}")
            raise
    
    def _generate_entity_note(self, entity: Entity, knowledge_graph: KnowledgeGraph,
                             document_content: Optional[DocumentContent] = None) -> ObsidianNote:
        """Generate a note for a single entity"""
        
        # Create filename
        filename = f"{entity.normalize_id()}{self.output_config.file_extension}"
        
        # Build frontmatter
        frontmatter = {}
        if self.output_config.include_yaml_frontmatter:
            frontmatter = {
                "type": entity.type,
                "aliases": entity.aliases,
                "created": datetime.now().isoformat(),
                "confidence": entity.confidence
            }
            
            # Add custom attributes
            if entity.attributes:
                frontmatter.update(entity.attributes)
            
            # Add source documents
            if entity.source_documents:
                frontmatter["sources"] = entity.source_documents
        
        # Build content
        content_parts = []
        
        # Add entity name as title (use simplified Chinese)
        content_parts.append(f"# {entity.get_simplified_name()}")
        content_parts.append("")
        
        # Add brief definition (LLM already includes [[]] links)
        if entity.brief_definition:
            content_parts.append("## 定义")
            content_parts.append(entity.get_simplified_brief_definition())
            content_parts.append("")
        
        # Get relations for this entity
        outgoing_relations, incoming_relations = knowledge_graph.get_entity_relations(entity.id)
        
        # Add description section with detailed content
        language = getattr(self.output_config, 'language', 'chs')
        if language == 'chs':
            content_parts.append("## 描述")
        else:
            content_parts.append("## Description")
        content_parts.append("")
        
        # Add the entity's detailed description if available (this may already contain [[]] links from LLM)
        if entity.description:
            # Get simplified description and process it to ensure proper double linking
            simplified_description = entity.get_simplified_description()
            processed_description = self._process_description_links(simplified_description, knowledge_graph, entity.id)
            content_parts.append(processed_description)
            content_parts.append("")
        
        # Add relation-based content if there are outgoing relations
        if outgoing_relations:
            # Group relations by type for better organization
            relation_groups = {}
            for relation in outgoing_relations:
                if relation.relation_type not in relation_groups:
                    relation_groups[relation.relation_type] = []
                relation_groups[relation.relation_type].append(relation)
            
            # Create subsections for each relation type
            for relation_type, relations in relation_groups.items():
                content_parts.append(f"### {self._format_relation_type(relation_type)}")
                for relation in relations:
                    target_entity = knowledge_graph.entities.get(relation.target_entity)
                    if target_entity:
                        # Create descriptive bullet points (LLM already includes [[]] links)
                        if relation.description:
                            # Convert relation description to simplified Chinese
                            import opencc
                            converter = opencc.OpenCC('t2s')
                            simplified_desc = converter.convert(relation.description)
                            content_parts.append(f"* **{self._format_relation_descriptor(relation_type)}**：{simplified_desc}，涉及[[{target_entity.get_simplified_name()}]] 。")
                        else:
                            content_parts.append(f"* 与[[{target_entity.get_simplified_name()}]]存在{self._format_relation_type(relation_type)}关系 。")
                content_parts.append("")
        
        # Add incoming relations (backlinks)
        if incoming_relations:
            language = getattr(self.output_config, 'language', 'chs')
            if language == 'chs':
                content_parts.append("## 被引用")
            else:
                content_parts.append("## Referenced By")
            content_parts.append("")
            
            relation_groups = {}
            for relation in incoming_relations:
                if relation.relation_type not in relation_groups:
                    relation_groups[relation.relation_type] = []
                relation_groups[relation.relation_type].append(relation)
            
            for relation_type, relations in relation_groups.items():
                content_parts.append(f"### {self._format_relation_type(relation_type)}")
                for relation in relations:
                    source_entity = knowledge_graph.entities.get(relation.source_entity)
                    if source_entity:
                        if relation.description:
                            # Convert relation description to simplified Chinese
                            import opencc
                            converter = opencc.OpenCC('t2s')
                            simplified_desc = converter.convert(relation.description)
                            content_parts.append(f"- [[{source_entity.get_simplified_name()}]] - {simplified_desc}")
                        else:
                            content_parts.append(f"- [[{source_entity.get_simplified_name()}]]")
                content_parts.append("")
        
        # Add embedded content (tables, formulas, images)
        if document_content:
            embedded_content = self._get_embedded_content_for_entity(entity, document_content)
            if embedded_content:
                language = getattr(self.output_config, 'language', 'chs')
                if language == 'chs':
                    content_parts.append("## 相关内容")
                else:
                    content_parts.append("## Related Content")
                content_parts.append("")
                content_parts.extend(embedded_content)
        
        # Add see also section with all related entities and semantic suggestions
        all_related_entity_ids = set()
        for relation in outgoing_relations:
            target = knowledge_graph.entities.get(relation.target_entity)
            if target:
                all_related_entity_ids.add(target.id)
        
        for relation in incoming_relations:
            source = knowledge_graph.entities.get(relation.source_entity)
            if source:
                all_related_entity_ids.add(source.id)
        
        # Add semantic suggestions based on entity type and name similarity
        entity_suggestions = self._find_semantic_links(entity, knowledge_graph, all_related_entity_ids)
        all_related_entity_ids.update(entity_suggestions)
        
        if all_related_entity_ids:
            language = getattr(self.output_config, 'language', 'chs')
            if language == 'chs':
                content_parts.append("## 相关链接")
            else:
                content_parts.append("## See Also")
            content_parts.append("")
            
            # Sort related entities alphabetically and convert to simplified Chinese
            simplified_entities = []
            for entity_id in all_related_entity_ids:
                related_entity = knowledge_graph.entities.get(entity_id)
                if related_entity:
                    simplified_entities.append(related_entity.get_simplified_name())
            
            # Sort and deduplicate
            sorted_entities = sorted(set(simplified_entities))
            for entity_name in sorted_entities:
                content_parts.append(f"- [[{entity_name}]]")
            content_parts.append("")
        
        # Create note
        note = ObsidianNote(
            filename=filename,
            title=entity.name,
            content="\n".join(content_parts),
            frontmatter=frontmatter,
            links=self._extract_links_from_content("\n".join(content_parts))
        )
        
        return note
    
    def _find_semantic_links(self, entity: Entity, knowledge_graph: KnowledgeGraph, 
                           existing_relations: Set[str]) -> Set[str]:
        """Find semantically related entities to create additional links"""
        suggestions = set()
        entity_lower = entity.get_simplified_name().lower()
        entity_words = set(entity_lower.split())
        
        # Skip if entity already has many relations
        if len(existing_relations) >= 8:
            return suggestions
        
        for other_entity in knowledge_graph.entities.values():
            if other_entity.id == entity.id:
                continue
            
            if other_entity.id in existing_relations:
                continue
            
            # Same type entities (moderate priority)
            if entity.type == other_entity.type and len(suggestions) < 3:
                suggestions.add(other_entity.id)
                continue
            
            # Name similarity (high priority)
            other_lower = other_entity.get_simplified_name().lower()
            other_words = set(other_lower.split())
            
            # Check for word overlap
            common_words = entity_words.intersection(other_words)
            if common_words and len(suggestions) < 5:
                suggestions.add(other_entity.id)
                continue
            
            # Check for substring matches
            if (entity_lower in other_lower or other_lower in entity_lower) and len(suggestions) < 5:
                suggestions.add(other_entity.id)
                continue
            
            # Description similarity (if available)
            if entity.description and other_entity.description:
                entity_desc_words = set(entity.get_simplified_description().lower().split())
                other_desc_words = set(other_entity.get_simplified_description().lower().split())
                desc_overlap = len(entity_desc_words.intersection(other_desc_words))
                
                if desc_overlap >= 2 and len(suggestions) < 6:
                    suggestions.add(other_entity.id)
        
        # Limit total suggestions to avoid overwhelming
        if len(suggestions) > 6:
            suggestions = set(list(suggestions)[:6])
        
        return suggestions
    
    def _generate_index_note(self, knowledge_graph: KnowledgeGraph, source_file: str) -> ObsidianNote:
        """Generate an index note for the entire knowledge graph"""
        
        frontmatter = {
            "type": "index",
            "created": datetime.now().isoformat(),
            "source": source_file,
            "entities": len(knowledge_graph.entities),
            "relations": len(knowledge_graph.relations)
        }
        
        content_parts = []
        
        # Add summary
        content_parts.append("## 知识图谱概览")
        content_parts.append("")
        content_parts.append(f"- **实体数量**: {len(knowledge_graph.entities)}")
        content_parts.append(f"- **关系数量**: {len(knowledge_graph.relations)}")
        content_parts.append(f"- **源文件**: {source_file}")
        content_parts.append("")
        
        # Group entities by type
        entities_by_type = {}
        for entity in knowledge_graph.entities.values():
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
        
        # Add entity listings by type
        content_parts.append("## 实体列表")
        content_parts.append("")
        
        for entity_type, entities in sorted(entities_by_type.items()):
            content_parts.append(f"### {entity_type}")
            content_parts.append("")
            
            # Sort entities alphabetically
            entities.sort(key=lambda e: e.name.lower())
            
            for entity in entities:
                simplified_name = entity.get_simplified_name()
                content_parts.append(f"- [[{simplified_name}]]")
                if entity.description:
                    # Add short description (first sentence) in simplified Chinese
                    simplified_desc = entity.get_simplified_description()
                    short_desc = simplified_desc.split('.')[0]
                    if len(short_desc) > 100:
                        short_desc = short_desc[:100] + "..."
                    content_parts.append(f"  - {short_desc}")
            
            content_parts.append("")
        
        # Add relation statistics
        content_parts.append("## 关系统计")
        content_parts.append("")
        
        relation_counts = {}
        for relation in knowledge_graph.relations:
            if relation.relation_type not in relation_counts:
                relation_counts[relation.relation_type] = 0
            relation_counts[relation.relation_type] += 1
        
        for relation_type, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
            content_parts.append(f"- **{self._format_relation_type(relation_type)}**: {count}")
        
        content_parts.append("")
        
        # Create index note
        note = ObsidianNote(
            filename=f"index{self.output_config.file_extension}",
            title="知识图谱索引",
            content="\n".join(content_parts),
            frontmatter=frontmatter,
            links=self._extract_links_from_content("\n".join(content_parts))
        )
        
        return note
    
    def _get_embedded_content_for_entity(self, entity: Entity, document_content: DocumentContent) -> List[str]:
        """Get embedded content (tables, formulas, images) related to an entity"""
        embedded_content = []
        
        # Search for entity mentions in tables
        if self.output_config.include_tables and document_content.tables:
            for table in document_content.tables:
                table_text = table.get('content', '') + ' ' + table.get('markdown', '')
                if entity.name.lower() in table_text.lower():
                    embedded_content.append("### 相关表格")
                    embedded_content.append("")
                    if table.get('markdown'):
                        embedded_content.append(table['markdown'])
                    elif table.get('html'):
                        embedded_content.append(table['html'])
                    embedded_content.append("")
        
        # Search for entity mentions in formulas
        if self.output_config.include_formulas and document_content.formulas:
            for formula in document_content.formulas:
                # Check if entity is mentioned near the formula (simple heuristic)
                if formula.get('latex'):
                    embedded_content.append("### 相关公式")
                    embedded_content.append("")
                    embedded_content.append(f"$${formula['latex']}$$")
                    embedded_content.append("")
        
        # Search for related images
        if self.output_config.include_images and document_content.images:
            for image in document_content.images:
                caption = image.get('caption', '')
                if entity.name.lower() in caption.lower():
                    embedded_content.append("### 相关图像")
                    embedded_content.append("")
                    
                    # Copy image to assets folder and create link
                    if image.get('path'):
                        asset_path = self._copy_image_to_assets(image['path'])
                        if asset_path:
                            embedded_content.append(f"![[{asset_path}]]")
                            if caption:
                                embedded_content.append(f"*{caption}*")
                    embedded_content.append("")
        
        return embedded_content
    
    def _copy_assets(self, document_content: DocumentContent):
        """Copy images and other assets to vault"""
        if not document_content.images:
            return
        
        # Create assets directory
        self.assets_path.mkdir(exist_ok=True)
        
        for image in document_content.images:
            if image.get('path'):
                self._copy_image_to_assets(image['path'])
    
    def _copy_image_to_assets(self, image_path: str) -> Optional[str]:
        """Copy an image to the assets folder and return relative path"""
        try:
            source_path = Path(image_path)
            if not source_path.exists():
                return None
            
            # Generate unique filename
            asset_filename = f"{source_path.stem}_{hash(str(source_path))}{source_path.suffix}"
            asset_path = self.assets_path / asset_filename
            
            # Copy file
            import shutil
            shutil.copy2(source_path, asset_path)
            
            # Return relative path for Obsidian
            return f"assets/{asset_filename}"
            
        except Exception as e:
            logger.warning(f"Failed to copy image {image_path}: {str(e)}")
            return None
    
    def _format_relation_type(self, relation_type: str) -> str:
        """Format relation type for display"""
        # Convert snake_case to Title Case
        formatted = relation_type.replace('_', ' ').title()
        
        # Special cases for common relations
        mappings = {
            "Defined As": "定义为",
            "Part Of": "属于",
            "Related To": "相关于",
            "Proposed By": "提出者",
            "Used In": "用于", 
            "Applies To": "适用于",
            "Causes": "导致",
            "Results In": "结果是",
            "Depends On": "依赖于",
            "Extends": "扩展"
        }
        
        return mappings.get(formatted, formatted)
    
    def _format_relation_descriptor(self, relation_type: str) -> str:
        """Format relation type as a descriptive phrase for bullet points"""
        # Convert snake_case to descriptive phrases
        mappings = {
            "defined_as": "定义特征",
            "part_of": "归属关系",
            "related_to": "相关联系",
            "proposed_by": "提出背景",
            "used_in": "应用场景",
            "applies_to": "适用范围",
            "causes": "因果关系",
            "results_in": "结果影响",
            "depends_on": "依赖关系",
            "extends": "扩展发展"
        }
        
        formatted = relation_type.replace('_', ' ').title()
        return mappings.get(relation_type.lower(), formatted)
    
    def _process_description_links(self, description: str, knowledge_graph: KnowledgeGraph, current_entity_id: str = None) -> str:
        """Process entity description to ensure proper double linking"""
        if not description:
            return description
        
        processed_description = description
        
        # Check if description already contains [[]] links
        existing_links = re.findall(r'\[\[([^\|\]]+)(?:\|[^\]]+)?\]\]', description)
        
        # If no existing links, try to add them automatically
        if not existing_links:
            # Find entity names mentioned in the description and add links
            for entity_id, entity in knowledge_graph.entities.items():
                # Skip the current entity to avoid self-linking
                if entity_id == current_entity_id:
                    continue
                
                simplified_name = entity.get_simplified_name()
                
                # Check if simplified name is mentioned in description
                if simplified_name.lower() in description.lower():
                    # Create a case-insensitive replacement pattern
                    pattern = re.compile(re.escape(simplified_name), re.IGNORECASE)
                    # Only replace if not already linked
                    if f"[[{simplified_name}]]" not in processed_description:
                        processed_description = pattern.sub(f"[[{simplified_name}]]", processed_description, count=1)
                
                # Also check original name in case it appears in description
                if entity.name.lower() in description.lower():
                    pattern = re.compile(re.escape(entity.name), re.IGNORECASE)
                    if f"[[{simplified_name}]]" not in processed_description:
                        processed_description = pattern.sub(f"[[{simplified_name}]]", processed_description, count=1)
                
                # Also check aliases
                for alias in entity.aliases:
                    # Convert alias to simplified Chinese
                    import opencc
                    converter = opencc.OpenCC('t2s')
                    simplified_alias = converter.convert(alias)
                    
                    if alias.lower() in description.lower():
                        pattern = re.compile(re.escape(alias), re.IGNORECASE)
                        if f"[[{simplified_name}]]" not in processed_description:
                            processed_description = pattern.sub(f"[[{simplified_name}]]", processed_description, count=1)
        
        return processed_description
    
    def _extract_links_from_content(self, content: str) -> List[str]:
        """Extract wiki-style links from content"""
        links = []
        
        # Find all [[link]] patterns
        link_pattern = r'\[\[([^\|\]]+)(?:\|[^\]]+)?\]\]'
        matches = re.findall(link_pattern, content)
        
        for match in matches:
            links.append(match)
        
        return list(set(links))  # Remove duplicates
    
    def _generate_graph_config(self):
        """Generate Obsidian graph view configuration"""
        config = {
            "collapse-filter": True,
            "search": "",
            "showTags": False,
            "showAttachments": False,
            "hideUnresolved": False,
            "showOrphans": True,
            "collapse-color-groups": True,
            "colorGroups": [],
            "collapse-display": False,
            "showArrow": True,
            "textFadeMultiplier": 0,
            "nodeSizeMultiplier": 1,
            "lineSizeMultiplier": 1,
            "collapse-forces": False,
            "centerStrength": 0.518713248970312,
            "repelStrength": 10,
            "linkStrength": 1,
            "linkDistance": 250,
            "scale": 1,
            "close": False
        }
        
        # Save graph configuration
        obsidian_dir = self.vault_path / ".obsidian"
        obsidian_dir.mkdir(exist_ok=True)
        
        graph_config_path = obsidian_dir / "graph.json"
        with open(graph_config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("Generated Obsidian graph configuration")
    
    def generate_statistics(self, knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
        """Generate statistics about the generated vault"""
        stats = {
            "total_notes": len(knowledge_graph.entities),
            "total_links": len(knowledge_graph.relations),
            "entity_types": {},
            "relation_types": {},
            "vault_path": str(self.vault_path),
            "assets_count": 0
        }
        
        # Count entity types
        for entity in knowledge_graph.entities.values():
            if entity.type not in stats["entity_types"]:
                stats["entity_types"][entity.type] = 0
            stats["entity_types"][entity.type] += 1
        
        # Count relation types
        for relation in knowledge_graph.relations:
            if relation.relation_type not in stats["relation_types"]:
                stats["relation_types"][relation.relation_type] = 0
            stats["relation_types"][relation.relation_type] += 1
        
        # Count assets
        if self.assets_path.exists():
            stats["assets_count"] = len(list(self.assets_path.glob("*")))
        
        return stats