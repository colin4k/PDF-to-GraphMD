"""
MinerU-based PDF parsing module
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import subprocess

from ..models import DocumentContent
from ..config import MinerUConfig


logger = logging.getLogger(__name__)


class MinerUParser:
    """PDF parser using MinerU library"""
    
    def __init__(self, config: MinerUConfig):
        self.config = config
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup MinerU environment and dependencies"""
        try:
            # Check if MinerU is available
            import magic_pdf
            logger.info("MinerU (magic-pdf) is available")
        except ImportError:
            logger.error("MinerU (magic-pdf) is not installed. Please install it first.")
            raise ImportError("MinerU is required but not installed")
    
    def parse_pdf(self, pdf_path: str) -> DocumentContent:
        """
        Parse PDF using MinerU and return structured content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentContent object with extracted content
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Starting MinerU parsing for: {pdf_path}")
            
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Run MinerU parsing
                result = self._run_mineru_extraction(pdf_path, temp_path)
                
                # Process results
                content = self._process_mineru_output(result, temp_path)
                
            logger.info(f"Successfully parsed PDF: {pdf_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise
    
    def _run_mineru_extraction(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run MinerU extraction process"""
        try:
            # Import MinerU components
            from magic_pdf.cli.magicpdf import do_parse
            from magic_pdf.config.drop_mode import DropMode
            from magic_pdf.config.make_content_config import MakeContentConfig
            
            # Configure MinerU parameters
            config_dict = {
                "input_path": str(pdf_path),
                "output_dir": str(output_dir),
                "output_format": self.config.output_formats,
                "parse_method": "auto",
                "lang": self.config.language,
                "apply_layout": True,
                "apply_formula": True,
                "apply_ocr": True,
            }
            
            # Enable GPU if available and configured
            if self.config.use_gpu and self._check_gpu_availability():
                config_dict["device"] = "cuda"
                logger.info("Using GPU acceleration for MinerU")
            else:
                config_dict["device"] = "cpu"
                logger.info("Using CPU for MinerU processing")
            
            # Execute MinerU parsing
            result = do_parse(
                pdf_path=str(pdf_path),
                output_dir=str(output_dir),
                method="auto",
                start_page_id=0,
                end_page_id=None,
                lang=self.config.language,
                layout_mode=True,
                formula_mode=True,
                table_mode=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"MinerU extraction failed: {str(e)}")
            # Fallback to command line interface if Python API fails
            return self._run_mineru_cli(pdf_path, output_dir)
    
    def _run_mineru_cli(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Fallback: Run MinerU via command line interface"""
        try:
            cmd = [
                "magic-pdf",
                "pdf-command-line",
                "--pdf", str(pdf_path),
                "--output-dir", str(output_dir),
                "--method", "auto"
            ]
            
            if self.config.language:
                cmd.extend(["--lang", self.config.language])
                
            if self.config.use_gpu and self._check_gpu_availability():
                cmd.append("--cuda")
            
            logger.info(f"Running MinerU CLI: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("MinerU CLI execution completed")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MinerU CLI failed: {e.stderr}")
            raise RuntimeError(f"MinerU CLI execution failed: {e.stderr}")
    
    def _process_mineru_output(self, result: Dict[str, Any], output_dir: Path) -> DocumentContent:
        """Process MinerU output and create DocumentContent"""
        try:
            # Find output files
            markdown_files = list(output_dir.glob("**/*.md"))
            json_files = list(output_dir.glob("**/*.json"))
            image_files = list(output_dir.glob("**/*.{png,jpg,jpeg,gif}"))
            
            # Read markdown content
            markdown_content = ""
            if markdown_files:
                with open(markdown_files[0], 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            # Read JSON data
            json_data = {}
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            
            # Extract text content from markdown (remove markdown formatting)
            text_content = self._extract_plain_text(markdown_content)
            
            # Extract structured elements
            images = self._extract_images_info(json_data, image_files)
            tables = self._extract_tables_info(json_data)
            formulas = self._extract_formulas_info(json_data)
            
            # Create DocumentContent object
            content = DocumentContent(
                text=text_content,
                markdown=markdown_content,
                json_data=json_data,
                images=images,
                tables=tables,
                formulas=formulas,
                metadata=self._extract_metadata(json_data)
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing MinerU output: {str(e)}")
            raise
    
    def _extract_plain_text(self, markdown: str) -> str:
        """Extract plain text from markdown content"""
        # Simple markdown to text conversion
        import re
        
        # Remove markdown formatting
        text = re.sub(r'#+\s*', '', markdown)  # Headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images
        
        return text.strip()
    
    def _extract_images_info(self, json_data: Dict, image_files: List[Path]) -> List[Dict[str, Any]]:
        """Extract image information from parsed data"""
        images = []
        
        # Extract from JSON data if available
        if 'images' in json_data:
            for img_data in json_data['images']:
                images.append({
                    'path': img_data.get('path', ''),
                    'caption': img_data.get('caption', ''),
                    'bbox': img_data.get('bbox', {}),
                    'page': img_data.get('page', 0)
                })
        
        # Add found image files
        for img_file in image_files:
            if not any(img['path'] == str(img_file) for img in images):
                images.append({
                    'path': str(img_file),
                    'caption': '',
                    'bbox': {},
                    'page': 0
                })
        
        return images
    
    def _extract_tables_info(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract table information from parsed data"""
        tables = []
        
        if 'tables' in json_data:
            for table_data in json_data['tables']:
                tables.append({
                    'content': table_data.get('content', ''),
                    'html': table_data.get('html', ''),
                    'markdown': table_data.get('markdown', ''),
                    'bbox': table_data.get('bbox', {}),
                    'page': table_data.get('page', 0)
                })
        
        return tables
    
    def _extract_formulas_info(self, json_data: Dict) -> List[Dict[str, Any]]:
        """Extract formula information from parsed data"""
        formulas = []
        
        if 'formulas' in json_data:
            for formula_data in json_data['formulas']:
                formulas.append({
                    'latex': formula_data.get('latex', ''),
                    'type': formula_data.get('type', 'inline'),  # inline or block
                    'bbox': formula_data.get('bbox', {}),
                    'page': formula_data.get('page', 0)
                })
        
        return formulas
    
    def _extract_metadata(self, json_data: Dict) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = {}
        
        if 'metadata' in json_data:
            metadata = json_data['metadata'].copy()
        
        # Add parsing information
        metadata.update({
            'parser': 'MinerU',
            'language': self.config.language,
            'gpu_used': self.config.use_gpu and self._check_gpu_availability()
        })
        
        return metadata
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for processing"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def parse_batch(self, pdf_paths: List[str]) -> List[DocumentContent]:
        """Parse multiple PDF files in batch"""
        results = []
        
        for pdf_path in pdf_paths:
            try:
                content = self.parse_pdf(pdf_path)
                results.append(content)
            except Exception as e:
                logger.error(f"Failed to parse {pdf_path}: {str(e)}")
                # Create empty content for failed parsing
                results.append(DocumentContent(
                    text="",
                    markdown="",
                    json_data={},
                    metadata={"error": str(e), "source": pdf_path}
                ))
        
        return results