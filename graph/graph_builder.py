"""
Knowledge graph construction and normalization module
"""
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import re
from difflib import SequenceMatcher

from ..models import Entity, Relation, KnowledgeGraph, KnowledgeTriple
from ..config import OntologyConfig


logger = logging.getLogger(__name__)


class GraphBuilder:
    """Knowledge graph construction and normalization"""
    
    def __init__(self, ontology_config: OntologyConfig):
        self.ontology_config = ontology_config
        self.entity_normalizer = EntityNormalizer()
        self.relation_validator = RelationValidator(ontology_config)
    
    def build_graph(self, extracted_graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Build unified knowledge graph from multiple extracted graphs
        
        Args:
            extracted_graphs: List of knowledge graphs from different sources/chunks
            
        Returns:
            Unified and normalized knowledge graph
        """
        try:
            logger.info(f"Building unified graph from {len(extracted_graphs)} source graphs")
            
            # Merge all graphs
            merged_graph = self._merge_graphs(extracted_graphs)
            
            # Normalize entities (resolve duplicates and inconsistencies)
            normalized_graph = self._normalize_entities(merged_graph)
            
            # Validate and clean relations
            cleaned_graph = self._clean_relations(normalized_graph)
            
            # Build final triples
            final_graph = self._build_triples(cleaned_graph)
            
            logger.info(f"Built unified graph with {len(final_graph.entities)} entities and {len(final_graph.relations)} relations")
            
            return final_graph
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise
    
    def _merge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """Merge multiple knowledge graphs into one"""
        merged_graph = KnowledgeGraph()
        
        for graph in graphs:
            # Merge entities
            for entity_id, entity in graph.entities.items():
                if entity_id not in merged_graph.entities:
                    merged_graph.entities[entity_id] = entity
                else:
                    # Merge entity information
                    existing_entity = merged_graph.entities[entity_id]
                    existing_entity = self._merge_entities(existing_entity, entity)
                    merged_graph.entities[entity_id] = existing_entity
            
            # Merge relations
            for relation in graph.relations:
                # Check for duplicate relations
                if not self._relation_exists(merged_graph.relations, relation):
                    merged_graph.relations.append(relation)
            
            # Merge triples
            merged_graph.triples.extend(graph.triples)
        
        return merged_graph
    
    def _merge_entities(self, entity1: Entity, entity2: Entity) -> Entity:
        """Merge two entities with the same ID"""
        merged_entity = Entity(
            id=entity1.id,
            name=entity1.name,
            type=entity1.type,
            description=entity1.description or entity2.description,
            aliases=list(set(entity1.aliases + entity2.aliases)),
            attributes={**entity1.attributes, **entity2.attributes},
            source_documents=list(set(entity1.source_documents + entity2.source_documents)),
            confidence=max(entity1.confidence, entity2.confidence)
        )
        return merged_entity
    
    def _relation_exists(self, relations: List[Relation], new_relation: Relation) -> bool:
        """Check if a relation already exists in the list"""
        for relation in relations:
            if (relation.source_entity == new_relation.source_entity and
                relation.relation_type == new_relation.relation_type and
                relation.target_entity == new_relation.target_entity):
                return True
        return False
    
    def _normalize_entities(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Normalize entities to resolve duplicates and inconsistencies"""
        logger.info("Normalizing entities...")
        
        # Find similar entities
        entity_clusters = self.entity_normalizer.find_entity_clusters(list(graph.entities.values()))
        
        # Create mapping from old IDs to normalized IDs
        id_mapping = {}
        normalized_entities = {}
        
        for cluster in entity_clusters:
            # Choose canonical entity for the cluster
            canonical_entity = self._choose_canonical_entity(cluster)
            
            # Generate normalized ID
            normalized_id = self._generate_normalized_id(canonical_entity)
            
            # Ensure uniqueness
            counter = 1
            original_id = normalized_id
            while normalized_id in normalized_entities:
                normalized_id = f"{original_id}_{counter}"
                counter += 1
            
            # Update canonical entity
            canonical_entity.id = normalized_id
            normalized_entities[normalized_id] = canonical_entity
            
            # Create mapping for all entities in the cluster
            for entity in cluster:
                id_mapping[entity.id] = normalized_id
        
        # Update relations with new entity IDs
        updated_relations = []
        for relation in graph.relations:
            updated_relation = Relation(
                source_entity=id_mapping.get(relation.source_entity, relation.source_entity),
                relation_type=relation.relation_type,
                target_entity=id_mapping.get(relation.target_entity, relation.target_entity),
                description=relation.description,
                confidence=relation.confidence,
                source_documents=relation.source_documents,
                attributes=relation.attributes
            )
            updated_relations.append(updated_relation)
        
        # Create normalized graph
        normalized_graph = KnowledgeGraph(
            entities=normalized_entities,
            relations=updated_relations,
            triples=[]  # Will be rebuilt
        )
        
        logger.info(f"Normalized {len(graph.entities)} entities to {len(normalized_entities)} unique entities")
        
        return normalized_graph
    
    def _choose_canonical_entity(self, cluster: List[Entity]) -> Entity:
        """Choose the canonical entity from a cluster of similar entities"""
        if len(cluster) == 1:
            return cluster[0]
        
        # Score entities based on completeness and confidence
        scored_entities = []
        for entity in cluster:
            score = (
                entity.confidence * 0.3 +
                len(entity.description) * 0.001 +
                len(entity.aliases) * 0.1 +
                len(entity.attributes) * 0.1 +
                (1.0 if entity.type in self.ontology_config.entity_types else 0.0) * 0.5
            )
            scored_entities.append((score, entity))
        
        # Return highest scoring entity
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        canonical = scored_entities[0][1]
        
        # Merge information from other entities in the cluster
        for _, entity in scored_entities[1:]:
            canonical = self._merge_entities(canonical, entity)
        
        return canonical
    
    def _generate_normalized_id(self, entity: Entity) -> str:
        """Generate normalized ID for entity"""
        # Use entity name as base
        normalized_id = entity.name.lower()
        
        # Replace spaces and special characters
        normalized_id = re.sub(r'[^\w\s-]', '', normalized_id)
        normalized_id = re.sub(r'\s+', '_', normalized_id)
        normalized_id = re.sub(r'_+', '_', normalized_id)
        normalized_id = normalized_id.strip('_')
        
        return normalized_id
    
    def _clean_relations(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Clean and validate relations"""
        logger.info("Cleaning relations...")
        
        cleaned_relations = []
        entity_ids = set(graph.entities.keys())
        
        for relation in graph.relations:
            # Validate relation
            if self.relation_validator.validate_relation(relation, entity_ids):
                cleaned_relations.append(relation)
            else:
                logger.debug(f"Removed invalid relation: {relation.source_entity} -> {relation.target_entity}")
        
        # Remove duplicate relations
        unique_relations = self._remove_duplicate_relations(cleaned_relations)
        
        # Create cleaned graph
        cleaned_graph = KnowledgeGraph(
            entities=graph.entities,
            relations=unique_relations,
            triples=[]
        )
        
        logger.info(f"Cleaned relations: {len(graph.relations)} -> {len(unique_relations)}")
        
        return cleaned_graph
    
    def _remove_duplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            relation_key = (
                relation.source_entity,
                relation.relation_type,
                relation.target_entity
            )
            
            if relation_key not in seen:
                seen.add(relation_key)
                unique_relations.append(relation)
            else:
                # Keep the relation with higher confidence
                existing_relation = next(
                    r for r in unique_relations 
                    if (r.source_entity, r.relation_type, r.target_entity) == relation_key
                )
                if relation.confidence > existing_relation.confidence:
                    unique_relations.remove(existing_relation)
                    unique_relations.append(relation)
        
        return unique_relations
    
    def _build_triples(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Build knowledge triples from entities and relations"""
        triples = []
        
        for relation in graph.relations:
            source_entity = graph.entities.get(relation.source_entity)
            target_entity = graph.entities.get(relation.target_entity)
            
            if source_entity and target_entity:
                triple = KnowledgeTriple(
                    subject=source_entity,
                    predicate=relation.relation_type,
                    object=target_entity,
                    confidence=relation.confidence,
                    source_text="",
                    source_document=""
                )
                triples.append(triple)
        
        # Update graph with triples
        graph.triples = triples
        
        return graph
    
    def validate_graph(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Validate the constructed knowledge graph"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {
                "entities": len(graph.entities),
                "relations": len(graph.relations),
                "triples": len(graph.triples),
                "entity_types": len(set(e.type for e in graph.entities.values())),
                "relation_types": len(set(r.relation_type for r in graph.relations))
            }
        }
        
        # Check for orphaned relations
        entity_ids = set(graph.entities.keys())
        orphaned_relations = []
        
        for relation in graph.relations:
            if relation.source_entity not in entity_ids:
                orphaned_relations.append(f"Source entity not found: {relation.source_entity}")
            if relation.target_entity not in entity_ids:
                orphaned_relations.append(f"Target entity not found: {relation.target_entity}")
        
        if orphaned_relations:
            validation_results["errors"].extend(orphaned_relations)
            validation_results["valid"] = False
        
        # Check entity types
        invalid_types = []
        valid_types = set(self.ontology_config.entity_types)
        
        for entity in graph.entities.values():
            if entity.type not in valid_types:
                invalid_types.append(f"Invalid entity type '{entity.type}' for entity '{entity.name}'")
        
        if invalid_types:
            validation_results["warnings"].extend(invalid_types)
        
        # Check relation types
        invalid_relations = []
        valid_relations = set(self.ontology_config.relation_types)
        
        for relation in graph.relations:
            if relation.relation_type not in valid_relations:
                invalid_relations.append(f"Invalid relation type: {relation.relation_type}")
        
        if invalid_relations:
            validation_results["warnings"].extend(invalid_relations)
        
        return validation_results


class EntityNormalizer:
    """Entity normalization and deduplication"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def find_entity_clusters(self, entities: List[Entity]) -> List[List[Entity]]:
        """Find clusters of similar entities"""
        clusters = []
        used_entities = set()
        
        for i, entity in enumerate(entities):
            if i in used_entities:
                continue
            
            cluster = [entity]
            used_entities.add(i)
            
            # Find similar entities
            for j, other_entity in enumerate(entities[i+1:], i+1):
                if j in used_entities:
                    continue
                
                if self._are_entities_similar(entity, other_entity):
                    cluster.append(other_entity)
                    used_entities.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_entities_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities are similar enough to be merged"""
        # Exact name match
        if entity1.name.lower() == entity2.name.lower():
            return True
        
        # Check aliases
        all_names1 = [entity1.name.lower()] + [alias.lower() for alias in entity1.aliases]
        all_names2 = [entity2.name.lower()] + [alias.lower() for alias in entity2.aliases]
        
        for name1 in all_names1:
            for name2 in all_names2:
                if name1 == name2:
                    return True
        
        # String similarity
        similarity = SequenceMatcher(None, entity1.name.lower(), entity2.name.lower()).ratio()
        if similarity >= self.similarity_threshold:
            return True
        
        # Check if one name is contained in the other (for abbreviations)
        if (entity1.name.lower() in entity2.name.lower() or 
            entity2.name.lower() in entity1.name.lower()):
            # Additional check: make sure it's not just a substring
            shorter_name = entity1.name if len(entity1.name) < len(entity2.name) else entity2.name
            longer_name = entity2.name if len(entity1.name) < len(entity2.name) else entity1.name
            
            if len(shorter_name) >= 3 and shorter_name.lower() in longer_name.lower():
                return True
        
        return False


class RelationValidator:
    """Relation validation and cleaning"""
    
    def __init__(self, ontology_config: OntologyConfig):
        self.ontology_config = ontology_config
        self.valid_relations = set(ontology_config.relation_types)
    
    def validate_relation(self, relation: Relation, valid_entity_ids: Set[str]) -> bool:
        """Validate a single relation"""
        # Check if source and target entities exist
        if relation.source_entity not in valid_entity_ids:
            return False
        
        if relation.target_entity not in valid_entity_ids:
            return False
        
        # Check if relation type is valid
        if relation.relation_type not in self.valid_relations:
            return False
        
        # Check for self-relations (entity pointing to itself)
        if relation.source_entity == relation.target_entity:
            return False
        
        # Check confidence threshold
        if relation.confidence < 0.1:
            return False
        
        return True