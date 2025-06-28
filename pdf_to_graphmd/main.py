"""
Main processing pipeline for PDF-to-GraphMD system
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .config import SystemConfig, ExtractionMethod, load_config
from .models import ProcessingResult, DocumentContent, KnowledgeGraph
from .parsers import MinerUParser
from .extractors import LLMExtractor, NLPExtractor  
from .graph import GraphBuilder
from .output import ObsidianGenerator
from .utils import (
    setup_logging, ProgressLogger, ErrorHandler, ErrorCategory, 
    ErrorSeverity, error_handler, log_performance
)


class PDFToGraphMDProcessor:
    """Main processor for the PDF-to-GraphMD pipeline"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(
            log_level=config.log_level,
            log_file=f"pdf_to_graphmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # Setup error handling
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("PDF-to-GraphMD processor initialized")
    
    def _initialize_components(self):
        """Initialize all processing components"""
        try:
            # PDF Parser
            self.pdf_parser = MinerUParser(self.config.mineru)
            
            # Knowledge Extractors
            if self.config.extraction_method == ExtractionMethod.LLM:
                self.knowledge_extractor = LLMExtractor(
                    self.config.llm, 
                    self.config.ontology,
                    self.config.output
                )
            else:
                self.knowledge_extractor = NLPExtractor(
                    self.config.nlp,
                    self.config.ontology
                )
            
            # Graph Builder
            self.graph_builder = GraphBuilder(self.config.ontology)
            
            # Output Generator
            self.output_generator = ObsidianGenerator(self.config.output)
            
            self.logger.info(f"Components initialized with {self.config.extraction_method.value} extraction method")
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL,
                context={"operation": "component_initialization"},
                suggested_action="Check configuration and dependencies"
            )
            raise
    
    @error_handler(ErrorCategory.PDF_PARSING, ErrorSeverity.HIGH, 
                  "Check PDF file integrity and MinerU installation")
    def process_single_pdf(self, pdf_path: str) -> ProcessingResult:
        """
        Process a single PDF file through the complete pipeline
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessingResult with complete processing information
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting processing of: {pdf_path}")
            
            # Stage 1: PDF Parsing
            self.logger.info("Stage 1: PDF Parsing and Content Extraction")
            document_content = self.pdf_parser.parse_pdf(pdf_path)
            
            # Stage 2: Knowledge Extraction  
            self.logger.info("Stage 2: Knowledge Extraction")
            knowledge_graph = self.knowledge_extractor.extract_knowledge(document_content)
            
            # Stage 3: Graph Construction and Normalization
            self.logger.info("Stage 3: Graph Construction and Normalization")
            normalized_graph = self.graph_builder.build_graph([knowledge_graph])
            
            # Stage 4: Output Generation
            self.logger.info("Stage 4: Obsidian Vault Generation")
            obsidian_notes = self.output_generator.generate_vault(
                normalized_graph, document_content, pdf_path
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create result
            result = ProcessingResult(
                source_file=pdf_path,
                success=True,
                document_content=document_content,
                knowledge_graph=normalized_graph,
                obsidian_notes=obsidian_notes,
                processing_time=processing_time
            )
            
            self.logger.info(f"Successfully processed {pdf_path} in {processing_time:.2f}s")
            self.logger.info(f"Generated {len(obsidian_notes)} notes with {len(normalized_graph.entities)} entities")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Handle error
            self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM, ErrorSeverity.HIGH,
                context={"pdf_path": pdf_path, "processing_time": processing_time}
            )
            
            # Return failed result
            result = ProcessingResult(
                source_file=pdf_path,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            
            return result
    
    def process_batch(self, pdf_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple PDF files
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of ProcessingResult objects
        """
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} files")
        
        progress = ProgressLogger(self.logger, len(pdf_paths), "Batch Processing")
        results = []
        
        for i, pdf_path in enumerate(pdf_paths):
            try:
                result = self.process_single_pdf(pdf_path)
                results.append(result)
                
                progress.update(1, f"Processed {pdf_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_path}: {str(e)}")
                
                # Create failed result
                failed_result = ProcessingResult(
                    source_file=pdf_path,
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)
                
                progress.update(1, f"Failed {pdf_path}")
        
        progress.complete(f"Processed {len(results)} files")
        
        # Log summary
        successful = len([r for r in results if r.success])
        failed = len(results) - successful
        
        self.logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        
        return results
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            self.config.validate()
            self.logger.info("Configuration validation passed")
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(str(e))
            self.logger.error(f"Configuration validation failed: {str(e)}")
        
        return validation_results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and error summary"""
        return {
            "error_summary": self.error_handler.get_error_summary(),
            "configuration": {
                "extraction_method": self.config.extraction_method.value,
                "output_directory": self.config.output.output_dir,
                "entity_types": len(self.config.ontology.entity_types),
                "relation_types": len(self.config.ontology.relation_types)
            }
        }


def create_processor(config_path: Optional[str] = None) -> PDFToGraphMDProcessor:
    """
    Create and configure a PDF-to-GraphMD processor
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured processor instance
    """
    config = load_config(config_path)
    return PDFToGraphMDProcessor(config)


def process_pdf_file(pdf_path: str, config_path: Optional[str] = None) -> ProcessingResult:
    """
    Convenience function to process a single PDF file
    
    Args:
        pdf_path: Path to PDF file
        config_path: Optional configuration file path
        
    Returns:
        ProcessingResult
    """
    processor = create_processor(config_path)
    return processor.process_single_pdf(pdf_path)


def process_pdf_directory(directory_path: str, config_path: Optional[str] = None) -> List[ProcessingResult]:
    """
    Convenience function to process all PDF files in a directory
    
    Args:
        directory_path: Path to directory containing PDF files
        config_path: Optional configuration file path
        
    Returns:
        List of ProcessingResult objects
    """
    directory = Path(directory_path)
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {directory_path}")
    
    processor = create_processor(config_path)
    return processor.process_batch([str(pdf) for pdf in pdf_files])