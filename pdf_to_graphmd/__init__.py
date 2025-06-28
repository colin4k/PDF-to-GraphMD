"""
PDF-to-GraphMD: Automated Knowledge Graph Construction System

Convert PDF documents into structured Obsidian-compatible knowledge graphs
using advanced document parsing and knowledge extraction techniques.
"""

__version__ = "1.0.0"
__author__ = "PDF-to-GraphMD Team"
__description__ = "Automated Knowledge Graph Construction System for PDF Documents"

from .main import (
    PDFToGraphMDProcessor,
    create_processor,
    process_pdf_file,
    process_pdf_directory
)

from .config import (
    SystemConfig,
    ExtractionMethod,
    load_config,
    load_default_config
)

from .models import (
    Entity,
    Relation,
    KnowledgeGraph,
    KnowledgeTriple,
    DocumentContent,
    ObsidianNote,
    ProcessingResult
)

__all__ = [
    # Main functionality
    'PDFToGraphMDProcessor',
    'create_processor', 
    'process_pdf_file',
    'process_pdf_directory',
    
    # Configuration
    'SystemConfig',
    'ExtractionMethod',
    'load_config',
    'load_default_config',
    
    # Data models
    'Entity',
    'Relation', 
    'KnowledgeGraph',
    'KnowledgeTriple',
    'DocumentContent',
    'ObsidianNote',
    'ProcessingResult'
]