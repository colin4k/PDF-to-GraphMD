"""
Configuration module for PDF-to-GraphMD system
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum
import yaml
import json


class ExtractionMethod(Enum):
    """Knowledge extraction method selection"""
    LLM = "llm"
    NLP = "nlp"


@dataclass
class MinerUConfig:
    """MinerU parsing configuration"""
    language: str = "ch"  # Language code for OCR
    use_gpu: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    batch_size: int = 1
    vlm_backend: str = "vlm-transformers"  # 支持 magic-pdf 2.0.x 的 VLM 后端
    source: str = "local"  # mineru CLI 的 --source 参数
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: Optional[str] = None  # If None, uses default ~/.cache/pdf_to_graphmd/mineru
    cache_expire_days: int = 30  # Cache expiration time in days


@dataclass
class LLMConfig:
    """LLM-based extraction configuration"""
    # API provider type
    api_provider: str = "openai"  # "openai" or "google"
    
    # OpenAI-compatible API settings
    model_name: str = "gemini-2.5-pro"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Google AI Studio settings
    google_model_name: str = "gemini-2.5-pro"
    google_api_key: Optional[str] = None
    
    # Common LLM settings
    temperature: float = 0.1
    max_tokens: int = 4000
    force_json: bool = True
    system_prompt: str = "You are an expert knowledge extraction system."
    max_retries: int = 3


@dataclass
class NLPConfig:
    """NLP-based extraction configuration"""
    spacy_model: str = "zh_core_web_sm"
    custom_ner_model: Optional[str] = None
    enable_custom_relations: bool = True


@dataclass
class OntologyConfig:
    """Knowledge ontology configuration"""
    entity_types: List[str] = field(default_factory=lambda: [
        "Person", "Organization", "Location", "Concept", "Theory", 
        "Method", "Tool", "Event", "Document", "Dataset"
    ])
    relation_types: List[str] = field(default_factory=lambda: [
        "defined_as", "part_of", "related_to", "proposed_by", "used_in",
        "applies_to", "causes", "results_in", "depends_on", "extends"
    ])


@dataclass
class OutputConfig:
    """Output generation configuration"""
    output_dir: str = "./obsidian_vault"
    file_extension: str = ".md"
    include_yaml_frontmatter: bool = True
    include_images: bool = True
    include_tables: bool = True
    include_formulas: bool = True
    language: str = "chs"
    language_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """Main system configuration"""
    extraction_method: ExtractionMethod = ExtractionMethod.LLM
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Processing options
    enable_incremental: bool = True
    max_workers: int = 4
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert extraction method string to enum
        if 'extraction_method' in config_dict:
            config_dict['extraction_method'] = ExtractionMethod(config_dict['extraction_method'])
        
        # Convert nested configs to dataclass objects
        if 'mineru' in config_dict:
            config_dict['mineru'] = MinerUConfig(**config_dict['mineru'])
        
        if 'llm' in config_dict:
            config_dict['llm'] = LLMConfig(**config_dict['llm'])
        
        if 'nlp' in config_dict:
            config_dict['nlp'] = NLPConfig(**config_dict['nlp'])
        
        if 'ontology' in config_dict:
            config_dict['ontology'] = OntologyConfig(**config_dict['ontology'])
        
        if 'output' in config_dict:
            config_dict['output'] = OutputConfig(**config_dict['output'])
            
        return cls(**config_dict)
    
    def to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = self.__dict__.copy()
        config_dict['extraction_method'] = self.extraction_method.value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self):
        """Validate configuration settings"""
        if self.extraction_method == ExtractionMethod.LLM:
            if self.llm.api_provider == "openai":
                if not self.llm.api_key and not os.getenv('OPENAI_API_KEY'):
                    raise ValueError("OpenAI-compatible API key must be provided")
            elif self.llm.api_provider == "google":
                if not self.llm.google_api_key and not os.getenv('GOOGLE_API_KEY'):
                    raise ValueError("Google API key must be provided")
            else:
                raise ValueError(f"Unsupported API provider: {self.llm.api_provider}")
        
        if not os.path.exists(os.path.dirname(self.output.output_dir)):
            os.makedirs(os.path.dirname(self.output.output_dir), exist_ok=True)


def load_default_config() -> SystemConfig:
    """Load default system configuration"""
    return SystemConfig()


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from file or use defaults"""
    if config_path and os.path.exists(config_path):
        return SystemConfig.from_file(config_path)
    return load_default_config()