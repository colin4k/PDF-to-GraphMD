"""
Command Line Interface for PDF-to-GraphMD system
"""
import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from .main import create_processor, process_pdf_file, process_pdf_directory
from .config import SystemConfig, ExtractionMethod, load_config
from .utils import setup_logging
from .parsers import MinerUParser


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="PDF-to-GraphMD: Convert PDF documents to Obsidian knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF file
  python -m pdf_to_graphmd --input document.pdf
  
  # Process multiple PDF files
  python -m pdf_to_graphmd --input file1.pdf file2.pdf file3.pdf
  
  # Process all PDFs in a directory
  python -m pdf_to_graphmd --input-dir ./documents/
  
  # Use custom configuration
  python -m pdf_to_graphmd --input document.pdf --config config.yaml
  
  # Use NLP extraction method
  python -m pdf_to_graphmd --input document.pdf --method nlp
  
  # Specify output directory
  python -m pdf_to_graphmd --input document.pdf --output ./my_vault/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        nargs="+",
        help="Input PDF file(s)"
    )
    input_group.add_argument(
        "--input-dir", "-d",
        help="Directory containing PDF files"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path (YAML)"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["llm", "nlp"],
        help="Knowledge extraction method (overrides config)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for Obsidian vault (overrides config)"
    )
    
    # LLM options
    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--llm-model",
        help="LLM model name (e.g., gpt-3.5-turbo)"
    )
    llm_group.add_argument(
        "--api-key",
        help="API key for LLM service"
    )
    llm_group.add_argument(
        "--base-url",
        help="Base URL for LLM API"
    )
    
    # Processing options
    parser.add_argument(
        "--language", 
        default="ch",
        help="Language for OCR processing (default: ch)"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch processing size"
    )
    
    # Output options
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Exclude images from output"
    )
    
    parser.add_argument(
        "--no-tables",
        action="store_true", 
        help="Exclude tables from output"
    )
    
    parser.add_argument(
        "--no-formulas",
        action="store_true",
        help="Exclude formulas from output"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    # Utility options
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--generate-config",
        help="Generate sample configuration file"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show processing statistics only (no actual processing)"
    )
    
    # Cache management options
    cache_group = parser.add_argument_group("Cache Management")
    cache_group.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cache information and exit"
    )
    cache_group.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached results and exit"
    )
    cache_group.add_argument(
        "--clear-cache-older-than",
        type=int,
        metavar="DAYS",
        help="Clear cache entries older than specified days"
    )
    cache_group.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable cache for this run"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="PDF-to-GraphMD 1.0.0"
    )
    
    return parser


def override_config_with_args(config: SystemConfig, args: argparse.Namespace) -> SystemConfig:
    """Override configuration with command line arguments"""
    
    # Extraction method
    if args.method:
        config.extraction_method = ExtractionMethod(args.method)
    
    # Output directory
    if args.output:
        config.output.output_dir = args.output
    
    # LLM settings
    if args.llm_model:
        config.llm.model_name = args.llm_model
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.base_url:
        config.llm.base_url = args.base_url
    
    # Processing settings
    if args.language:
        config.mineru.language = args.language
    if args.use_gpu:
        config.mineru.use_gpu = True
    if args.batch_size:
        config.mineru.batch_size = args.batch_size
    
    # Output settings
    if args.no_images:
        config.output.include_images = False
    if args.no_tables:
        config.output.include_tables = False
    if args.no_formulas:
        config.output.include_formulas = False
    
    # Logging
    if args.log_level:
        config.log_level = args.log_level
    
    # Cache settings
    if args.disable_cache:
        config.mineru.enable_cache = False
    
    return config


def generate_sample_config(output_path: str):
    """Generate a sample configuration file"""
    
    config = SystemConfig()
    config.to_file(output_path)
    
    print(f"Sample configuration generated: {output_path}")
    print("Edit this file to customize your settings.")


def validate_configuration(config_path: Optional[str]) -> bool:
    """Validate configuration file"""
    
    try:
        config = load_config(config_path)
        processor = create_processor(config_path)
        
        validation_result = processor.validate_configuration()
        
        if validation_result["valid"]:
            print("✓ Configuration is valid")
            return True
        else:
            print("✗ Configuration validation failed:")
            for error in validation_result["errors"]:
                print(f"  - {error}")
            for warning in validation_result["warnings"]:
                print(f"  ! {warning}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration validation failed: {str(e)}")
        return False


def print_processing_stats(results: List):
    """Print processing statistics"""
    
    if not results:
        print("No results to display")
        return
    
    successful = len([r for r in results if r.success])
    failed = len(results) - successful
    
    print(f"\\n📊 Processing Statistics:")
    print(f"  Total files: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        total_entities = sum(len(r.knowledge_graph.entities) for r in results if r.success)
        total_relations = sum(len(r.knowledge_graph.relations) for r in results if r.success)
        total_notes = sum(len(r.obsidian_notes) for r in results if r.success)
        avg_time = sum(r.processing_time for r in results if r.success) / successful
        
        print(f"  Total entities: {total_entities}")
        print(f"  Total relations: {total_relations}")
        print(f"  Total notes: {total_notes}")
        print(f"  Average processing time: {avg_time:.2f}s")
    
    # Show failed files
    if failed > 0:
        print(f"\\n❌ Failed files:")
        for result in results:
            if not result.success:
                print(f"  - {result.source_file}: {result.error_message}")


def handle_cache_commands(args: argparse.Namespace, config: SystemConfig) -> bool:
    """
    Handle cache-related commands
    
    Args:
        args: Command line arguments
        config: System configuration
        
    Returns:
        True if a cache command was handled (and program should exit)
    """
    parser = MinerUParser(config.mineru)
    
    if args.cache_info:
        cache_info = parser.get_cache_info()
        print("🗄️  Cache Information:")
        print(f"  Cache directory: {cache_info['cache_dir']}")
        print(f"  Cache exists: {cache_info['exists']}")
        
        if cache_info['exists']:
            print(f"  Number of cached files: {cache_info['file_count']}")
            print(f"  Total cache size: {cache_info['total_size_mb']:.2f} MB")
            
            if cache_info['file_count'] > 0:
                print("  Cached files:")
                for file_info in cache_info['files']:
                    print(f"    - {file_info['name']}: {file_info['size_mb']:.2f} MB (modified: {file_info['modified']})")
        
        return True
    
    if args.clear_cache:
        print("🗑️  Clearing all cache...")
        parser.clear_cache()
        print("✓ Cache cleared successfully")
        return True
    
    if args.clear_cache_older_than is not None:
        days = args.clear_cache_older_than
        print(f"🗑️  Clearing cache older than {days} days...")
        parser.clear_cache(older_than_days=days)
        print(f"✓ Cache entries older than {days} days cleared successfully")
        return True
    
    return False


def main():
    """Main CLI entry point"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    try:
        # Handle utility commands
        if args.generate_config:
            generate_sample_config(args.generate_config)
            return 0
        
        if args.validate_config:
            if validate_configuration(args.config):
                return 0
            else:
                return 1
        
        # Load and override configuration
        config = load_config(args.config)
        config = override_config_with_args(config, args)
        
        # Handle cache commands
        if handle_cache_commands(args, config):
            return 0
        
        # Show configuration if stats only
        if args.stats_only:
            print("📋 Current Configuration:")
            print(f"  Extraction method: {config.extraction_method.value}")
            print(f"  Output directory: {config.output.output_dir}")
            print(f"  Entity types: {len(config.ontology.entity_types)}")
            print(f"  Relation types: {len(config.ontology.relation_types)}")
            return 0
        
        # Process files
        print("🚀 Starting PDF-to-GraphMD processing...")
        
        results = []
        
        if args.input:
            # Process individual files
            for pdf_path in args.input:
                print(f"📄 Processing: {pdf_path}")
                result = process_pdf_file(pdf_path, args.config)
                results.append(result)
                
                if result.success:
                    print(f"✓ Successfully processed {pdf_path}")
                    print(f"  Generated {len(result.obsidian_notes)} notes")
                    print(f"  Extracted {len(result.knowledge_graph.entities)} entities")
                else:
                    print(f"✗ Failed to process {pdf_path}: {result.error_message}")
        
        elif args.input_dir:
            # Process directory
            print(f"📁 Processing directory: {args.input_dir}")
            results = process_pdf_directory(args.input_dir, args.config)
        
        # Print final statistics
        print_processing_stats(results)
        
        # Determine exit code
        failed_count = len([r for r in results if not r.success])
        if failed_count == 0:
            print(f"\\n🎉 All files processed successfully!")
            print(f"Obsidian vault created at: {config.output.output_dir}")
            return 0
        elif failed_count < len(results):
            print(f"\\n⚠️  Partial success: {failed_count} files failed")
            return 2
        else:
            print(f"\\n💥 All files failed to process")
            return 1
    
    except KeyboardInterrupt:
        print("\\n⚠️  Processing interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"💥 Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())