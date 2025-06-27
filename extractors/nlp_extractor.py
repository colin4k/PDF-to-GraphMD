"""
NLP-based knowledge extraction module using spaCy
"""
import logging
from typing import Dict, List, Optional, Tuple, Set
import re
from collections import defaultdict

from ..models import Entity, Relation, KnowledgeTriple, KnowledgeGraph, DocumentContent
from ..config import NLPConfig, OntologyConfig


logger = logging.getLogger(__name__)


class NLPExtractor:
    """spaCy-based knowledge extraction with custom NER and relation extraction"""
    
    def __init__(self, nlp_config: NLPConfig, ontology_config: OntologyConfig):
        self.nlp_config = nlp_config
        self.ontology_config = ontology_config
        self.nlp = None
        self.matcher = None
        self._setup_spacy()
    
    def _setup_spacy(self):
        """Setup spaCy pipeline and components"""
        try:
            import spacy
            from spacy.matcher import Matcher, DependencyMatcher
            
            # Load spaCy model
            try:
                self.nlp = spacy.load(self.nlp_config.spacy_model)
                logger.info(f"Loaded spaCy model: {self.nlp_config.spacy_model}")
            except OSError:
                logger.warning(f"Model {self.nlp_config.spacy_model} not found, using default")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Setup matchers
            self.matcher = Matcher(self.nlp.vocab)
            self.dep_matcher = DependencyMatcher(self.nlp.vocab)
            
            # Add custom patterns
            self._add_custom_patterns()
            
            # Load custom NER model if specified
            if self.nlp_config.custom_ner_model:
                self._load_custom_ner_model()
            
        except ImportError:
            logger.error("spaCy not installed. Please install: pip install spacy")
            raise
        except Exception as e:
            logger.error(f"Error setting up spaCy: {str(e)}")
            raise
    
    def _add_custom_patterns(self):
        """Add custom patterns for entity and relation extraction"""
        
        # Entity patterns
        concept_patterns = [
            [{"LOWER": {"IN": ["theory", "concept", "principle", "law", "theorem"]}},
             {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
            [{"POS": "PROPN", "OP": "+"}, 
             {"LOWER": {"IN": ["theory", "concept", "principle", "law", "theorem"]}}]
        ]
        
        method_patterns = [
            [{"LOWER": {"IN": ["method", "algorithm", "technique", "approach", "process"]}},
             {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
            [{"POS": "PROPN", "OP": "+"}, 
             {"LOWER": {"IN": ["method", "algorithm", "technique", "approach"]}}]
        ]
        
        # Add patterns to matcher
        self.matcher.add("CONCEPT", concept_patterns)
        self.matcher.add("METHOD", method_patterns)
        
        # Relation patterns for dependency matcher
        definition_patterns = [
            {
                "RIGHT_ID": "definition",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["define", "refer", "mean", "denote"]}},
                "LEFT_ID": "subject",
                "LEFT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
                "REL_OP": ">"
            }
        ]
        
        causation_patterns = [
            {
                "RIGHT_ID": "cause",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["cause", "lead", "result", "produce"]}},
                "LEFT_ID": "subject",
                "LEFT_ATTRS": {"DEP": "nsubj"},
                "REL_OP": ">"
            }
        ]
        
        self.dep_matcher.add("DEFINITION", definition_patterns)
        self.dep_matcher.add("CAUSATION", causation_patterns)
    
    def _load_custom_ner_model(self):
        """Load custom NER model if available"""
        try:
            import spacy
            
            custom_nlp = spacy.load(self.nlp_config.custom_ner_model)
            
            # Add custom NER component to main pipeline
            self.nlp.add_pipe("custom_ner", source=custom_nlp)
            logger.info(f"Loaded custom NER model: {self.nlp_config.custom_ner_model}")
            
        except Exception as e:
            logger.warning(f"Could not load custom NER model: {str(e)}")
    
    def extract_knowledge(self, content: DocumentContent) -> KnowledgeGraph:
        """
        Extract knowledge graph from document content using NLP
        
        Args:
            content: Parsed document content
            
        Returns:
            KnowledgeGraph with extracted entities and relations
        """
        try:
            logger.info("Starting NLP-based knowledge extraction")
            
            # Process text with spaCy
            doc = self.nlp(content.text)
            
            # Extract entities
            entities = self._extract_entities(doc)
            
            # Extract relations
            relations = self._extract_relations(doc, entities)
            
            # Create knowledge graph
            kg = KnowledgeGraph()
            
            # Add entities
            for entity in entities:
                kg.add_entity(entity)
            
            # Add relations
            for relation in relations:
                kg.add_relation(relation)
            
            # Create triples
            entity_dict = {e.id: e for e in entities}
            for relation in relations:
                if relation.source_entity in entity_dict and relation.target_entity in entity_dict:
                    triple = KnowledgeTriple(
                        subject=entity_dict[relation.source_entity],
                        predicate=relation.relation_type,
                        object=entity_dict[relation.target_entity],
                        confidence=relation.confidence,
                        source_text="",  # Could be improved to track source
                        source_document=""
                    )
                    kg.add_triple(triple)
            
            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations")
            return kg
            
        except Exception as e:
            logger.error(f"Error in NLP knowledge extraction: {str(e)}")
            raise
    
    def _extract_entities(self, doc) -> List[Entity]:
        """Extract entities using NER and pattern matching"""
        entities = []
        seen_entities = set()
        
        # Extract named entities
        for ent in doc.ents:
            entity_type = self._map_spacy_label_to_ontology(ent.label_)
            if entity_type and ent.text.lower() not in seen_entities:
                
                entity = Entity(
                    id=self._generate_entity_id(ent.text),
                    name=ent.text,
                    type=entity_type,
                    description=self._extract_entity_description(ent, doc),
                    aliases=self._find_entity_aliases(ent, doc),
                    confidence=0.8  # Base confidence for NER
                )
                entities.append(entity)
                seen_entities.add(ent.text.lower())
        
        # Extract entities using custom patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            match_label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            if span.text.lower() not in seen_entities:
                entity_type = self._map_pattern_to_ontology(match_label)
                
                entity = Entity(
                    id=self._generate_entity_id(span.text),
                    name=span.text,
                    type=entity_type,
                    description=self._extract_entity_description(span, doc),
                    aliases=[],
                    confidence=0.7  # Lower confidence for pattern matching
                )
                entities.append(entity)
                seen_entities.add(span.text.lower())
        
        return entities
    
    def _extract_relations(self, doc, entities: List[Entity]) -> List[Relation]:
        """Extract relations using dependency parsing and pattern matching"""
        relations = []
        entity_spans = self._create_entity_span_map(doc, entities)
        
        # Extract relations using dependency matcher
        dep_matches = self.dep_matcher(doc)
        for match_id, matches in dep_matches:
            match_label = self.nlp.vocab.strings[match_id]
            
            for match in matches:
                relation_type = self._map_dependency_to_relation(match_label)
                
                # Find entities involved in the relation
                source_token = doc[match[0]]
                target_token = doc[match[1]]
                
                source_entity = self._find_entity_for_token(source_token, entity_spans)
                target_entity = self._find_entity_for_token(target_token, entity_spans)
                
                if source_entity and target_entity:
                    relation = Relation(
                        source_entity=source_entity.id,
                        relation_type=relation_type,
                        target_entity=target_entity.id,
                        description=f"{source_entity.name} {relation_type} {target_entity.name}",
                        confidence=0.6  # Moderate confidence for dependency relations
                    )
                    relations.append(relation)
        
        # Extract relations using rule-based patterns
        rule_relations = self._extract_rule_based_relations(doc, entities)
        relations.extend(rule_relations)
        
        return relations
    
    def _extract_rule_based_relations(self, doc, entities: List[Entity]) -> List[Relation]:
        """Extract relations using rule-based patterns"""
        relations = []
        entity_dict = {e.name.lower(): e for e in entities}
        
        # Pattern: "X is defined as Y" or "X refers to Y"
        definition_patterns = [
            r"(.+?)\s+(?:is|are)\s+defined\s+as\s+(.+?)(?:\.|,|;|$)",
            r"(.+?)\s+refers?\s+to\s+(.+?)(?:\.|,|;|$)",
            r"(.+?)\s+means?\s+(.+?)(?:\.|,|;|$)"
        ]
        
        for pattern in definition_patterns:
            for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                source_entity = self._find_matching_entity(source_text, entity_dict)
                target_entity = self._find_matching_entity(target_text, entity_dict)
                
                if source_entity and target_entity:
                    relation = Relation(
                        source_entity=source_entity.id,
                        relation_type="defined_as",
                        target_entity=target_entity.id,
                        description=f"{source_entity.name} is defined as {target_entity.name}",
                        confidence=0.7
                    )
                    relations.append(relation)
        
        # Pattern: "X causes Y" or "X leads to Y"
        causation_patterns = [
            r"(.+?)\s+causes?\s+(.+?)(?:\.|,|;|$)",
            r"(.+?)\s+leads?\s+to\s+(.+?)(?:\.|,|;|$)",
            r"(.+?)\s+results?\s+in\s+(.+?)(?:\.|,|;|$)"
        ]
        
        for pattern in causation_patterns:
            for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                source_entity = self._find_matching_entity(source_text, entity_dict)
                target_entity = self._find_matching_entity(target_text, entity_dict)
                
                if source_entity and target_entity:
                    relation = Relation(
                        source_entity=source_entity.id,
                        relation_type="causes",
                        target_entity=target_entity.id,
                        description=f"{source_entity.name} causes {target_entity.name}",
                        confidence=0.6
                    )
                    relations.append(relation)
        
        return relations
    
    def _map_spacy_label_to_ontology(self, spacy_label: str) -> Optional[str]:
        """Map spaCy NER labels to ontology entity types"""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization", 
            "GPE": "Location",
            "NORP": "Organization",
            "EVENT": "Event",
            "WORK_OF_ART": "Document",
            "LAW": "Theory",
            "LANGUAGE": "Concept",
            "PRODUCT": "Tool"
        }
        return mapping.get(spacy_label)
    
    def _map_pattern_to_ontology(self, pattern_label: str) -> str:
        """Map custom pattern labels to ontology entity types"""
        mapping = {
            "CONCEPT": "Concept",
            "METHOD": "Method",
            "THEORY": "Theory",
            "TOOL": "Tool"
        }
        return mapping.get(pattern_label, "Concept")
    
    def _map_dependency_to_relation(self, dep_label: str) -> str:
        """Map dependency patterns to relation types"""
        mapping = {
            "DEFINITION": "defined_as",
            "CAUSATION": "causes",
            "COMPARISON": "related_to",
            "POSSESSION": "part_of"
        }
        return mapping.get(dep_label, "related_to")
    
    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate unique entity ID from name"""
        # Clean and normalize the name
        entity_id = re.sub(r'[^\w\s-]', '', entity_name)
        entity_id = re.sub(r'\s+', '_', entity_id.strip())
        return entity_id.lower()
    
    def _extract_entity_description(self, entity_span, doc) -> str:
        """Extract description for an entity from surrounding context"""
        # Look for sentences containing the entity
        for sent in doc.sents:
            if entity_span.start >= sent.start and entity_span.end <= sent.end:
                # Return the sentence as description
                return sent.text.strip()
        return ""
    
    def _find_entity_aliases(self, entity_span, doc) -> List[str]:
        """Find alternative names/aliases for an entity"""
        aliases = []
        entity_text = entity_span.text.lower()
        
        # Look for appositive constructions
        for token in doc:
            if (token.dep_ == "appos" and 
                token.head.text.lower() == entity_text):
                aliases.append(token.text)
        
        return aliases
    
    def _create_entity_span_map(self, doc, entities: List[Entity]) -> Dict:
        """Create mapping from token positions to entities"""
        entity_spans = {}
        
        for entity in entities:
            for token in doc:
                if token.text.lower() in entity.name.lower():
                    entity_spans[token.i] = entity
        
        return entity_spans
    
    def _find_entity_for_token(self, token, entity_spans: Dict) -> Optional[Entity]:
        """Find entity associated with a token"""
        return entity_spans.get(token.i)
    
    def _find_matching_entity(self, text: str, entity_dict: Dict[str, Entity]) -> Optional[Entity]:
        """Find entity that matches the given text"""
        text_lower = text.lower()
        
        # Exact match
        if text_lower in entity_dict:
            return entity_dict[text_lower]
        
        # Partial match
        for entity_name, entity in entity_dict.items():
            if entity_name in text_lower or text_lower in entity_name:
                return entity
        
        return None
    
    def validate_extraction(self, kg: KnowledgeGraph) -> Dict[str, any]:
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
        
        return validation_results