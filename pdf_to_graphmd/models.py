"""
Data models for PDF-to-GraphMD system
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
from pathlib import Path


@dataclass
class DocumentContent:
    """Parsed document content from MinerU"""
    text: str
    markdown: str
    json_data: Dict[str, Any]
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    name: str
    type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_documents: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def normalize_id(self) -> str:
        """Generate normalized entity ID for filename"""
        # Replace special characters and spaces
        normalized = self.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove other problematic characters
        normalized = "".join(c for c in normalized if c.isalnum() or c in "_-.")
        return normalized


@dataclass
class Relation:
    """Knowledge graph relation/edge"""
    source_entity: str
    relation_type: str
    target_entity: str
    description: str = ""
    confidence: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeTriple:
    """Knowledge graph triple (subject-predicate-object)"""
    subject: Entity
    predicate: str
    object: Union[Entity, str]
    confidence: float = 1.0
    source_text: str = ""
    source_document: str = ""


@dataclass
class KnowledgeGraph:
    """Complete knowledge graph representation"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    triples: List[KnowledgeTriple] = field(default_factory=list)
    
    def add_entity(self, entity: Entity):
        """Add entity to the graph"""
        self.entities[entity.id] = entity
    
    def add_relation(self, relation: Relation):
        """Add relation to the graph"""
        self.relations.append(relation)
    
    def add_triple(self, triple: KnowledgeTriple):
        """Add triple to the graph"""
        self.triples.append(triple)
    
    def get_entity_relations(self, entity_id: str) -> Tuple[List[Relation], List[Relation]]:
        """Get outgoing and incoming relations for an entity"""
        outgoing = [r for r in self.relations if r.source_entity == entity_id]
        incoming = [r for r in self.relations if r.target_entity == entity_id]
        return outgoing, incoming
    
    def normalize_entities(self):
        """Normalize entity IDs and resolve duplicates"""
        normalized_entities = {}
        entity_mapping = {}
        
        for entity_id, entity in self.entities.items():
            normalized_id = entity.normalize_id()
            
            # Handle duplicate normalized IDs
            counter = 1
            original_normalized_id = normalized_id
            while normalized_id in normalized_entities:
                normalized_id = f"{original_normalized_id}_{counter}"
                counter += 1
            
            entity.id = normalized_id
            normalized_entities[normalized_id] = entity
            entity_mapping[entity_id] = normalized_id
        
        # Update relations with new entity IDs
        for relation in self.relations:
            if relation.source_entity in entity_mapping:
                relation.source_entity = entity_mapping[relation.source_entity]
            if relation.target_entity in entity_mapping:
                relation.target_entity = entity_mapping[relation.target_entity]
        
        self.entities = normalized_entities


@dataclass
class ObsidianNote:
    """Obsidian-compatible markdown note"""
    filename: str
    title: str
    content: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate complete markdown content with frontmatter"""
        lines = []
        
        # Add YAML frontmatter
        if self.frontmatter:
            lines.append("---")
            for key, value in self.frontmatter.items():
                if isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("---")
            lines.append("")
        
        # Add title
        lines.append(f"# {self.title}")
        lines.append("")
        
        # Add content
        lines.append(self.content)
        
        return "\n".join(lines)
    
    def save(self, output_dir: Path):
        """Save note to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / self.filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())


@dataclass
class ProcessingResult:
    """Result of processing a PDF document"""
    source_file: str
    success: bool
    document_content: Optional[DocumentContent] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    obsidian_notes: List[ObsidianNote] = field(default_factory=list)
    error_message: str = ""
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_file": self.source_file,
            "success": self.success,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "entity_count": len(self.knowledge_graph.entities) if self.knowledge_graph else 0,
            "relation_count": len(self.knowledge_graph.relations) if self.knowledge_graph else 0,
            "note_count": len(self.obsidian_notes)
        }