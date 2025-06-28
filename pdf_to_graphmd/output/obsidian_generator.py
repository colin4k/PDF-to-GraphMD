"""
Obsidian-compatible Markdown file generation module
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
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
        
        # Add entity name as title
        content_parts.append(f"# {entity.name}")
        content_parts.append("")
        
        # Add description
        if entity.description:
            content_parts.append(entity.description)
            content_parts.append("")
        
        # Get relations for this entity
        outgoing_relations, incoming_relations = knowledge_graph.get_entity_relations(entity.id)
        
        # Add outgoing relations with language-aware headers
        if outgoing_relations:
            language = getattr(self.output_config, 'language', 'en')
            if language == 'zh':
                content_parts.append("## 相关链接")
            else:
                content_parts.append("## Related Links")
            content_parts.append("")
            
            # Group relations by type
            relation_groups = {}
            for relation in outgoing_relations:
                if relation.relation_type not in relation_groups:
                    relation_groups[relation.relation_type] = []
                relation_groups[relation.relation_type].append(relation)
            
            for relation_type, relations in relation_groups.items():
                content_parts.append(f"### {self._format_relation_type(relation_type)}")
                for relation in relations:
                    target_entity = knowledge_graph.entities.get(relation.target_entity)
                    if target_entity:
                        content_parts.append(f"- [[{target_entity.normalize_id()}|{target_entity.name}]]")
                        if relation.description:
                            content_parts.append(f"  - {relation.description}")
                content_parts.append("")
        
        # Add incoming relations (backlinks)
        if incoming_relations:
            language = getattr(self.output_config, 'language', 'en')
            if language == 'zh':
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
                        content_parts.append(f"- [[{source_entity.normalize_id()}|{source_entity.name}]]")
                content_parts.append("")
        
        # Add embedded content (tables, formulas, images)
        if document_content:
            embedded_content = self._get_embedded_content_for_entity(entity, document_content)
            if embedded_content:
                language = getattr(self.output_config, 'language', 'en')
                if language == 'zh':
                    content_parts.append("## 相关内容")
                else:
                    content_parts.append("## Related Content")
                content_parts.append("")
                content_parts.extend(embedded_content)
        
        # Create note
        note = ObsidianNote(
            filename=filename,
            title=entity.name,
            content="\\n".join(content_parts),
            frontmatter=frontmatter,
            links=self._extract_links_from_content("\\n".join(content_parts))
        )
        
        return note
    
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
                content_parts.append(f"- [[{entity.normalize_id()}|{entity.name}]]")
                if entity.description:
                    # Add short description (first sentence)
                    short_desc = entity.description.split('.')[0]
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
            content="\\n".join(content_parts),
            frontmatter=frontmatter,
            links=self._extract_links_from_content("\\n".join(content_parts))
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