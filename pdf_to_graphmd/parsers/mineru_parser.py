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
import shlex
import hashlib
import pickle
from datetime import datetime

from ..models import DocumentContent
from ..config import MinerUConfig


logger = logging.getLogger(__name__)


class MinerUParser:
    """PDF parser using MinerU library"""
    
    def __init__(self, config: MinerUConfig):
        self.config = config
        
        # Setup cache directory
        if config.cache_dir:
            self.cache_dir = Path(config.cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "pdf_to_graphmd" / "mineru"
        
        if config.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup MinerU environment and dependencies"""
        try:
            # Check if mineru CLI is available
            result = subprocess.run(
                ["mineru", "--help"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info("MinerU CLI is available")
            else:
                raise FileNotFoundError("mineru command not found")
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.error("MinerU CLI is not installed or not in PATH. Please install mineru first.")
            raise ImportError("MinerU CLI is required but not available")
    
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
            
            # Check cache first (if enabled)
            if self.config.enable_cache:
                cache_key = self._generate_cache_key(pdf_path)
                cached_content = self._load_from_cache(cache_key)
                
                if cached_content:
                    logger.info(f"Found cached result for: {pdf_path}, skipping MinerU execution")
                    return cached_content
            
            logger.info(f"Starting MinerU parsing for: {pdf_path}")
            
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Run MinerU parsing
                result = self._run_mineru_extraction(pdf_path, temp_path)
                
                # Process results
                content = self._process_mineru_output(result, temp_path)
                
                # Cache the result (if enabled)
                if self.config.enable_cache:
                    self._save_to_cache(cache_key, content)
                
            logger.info(f"Successfully parsed PDF: {pdf_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise
    
    def _run_mineru_extraction(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run MinerU extraction process"""
        try:
            # Use mineru CLI directly
            logger.info("Using MinerU CLI interface")
            return self._run_mineru_cli(pdf_path, output_dir)
            
        except Exception as e:
            logger.error(f"MinerU extraction failed: {str(e)}")
            raise
    
    def _run_mineru_cli(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Fallback: Run MinerU via command line interface"""
        try:
            # 确保路径参数正确处理，特别是包含空格的文件名
            cmd = [
                "mineru",
                "-p", str(pdf_path),
                "-o", str(output_dir)
            ]

            # VLM backend
            if hasattr(self.config, 'vlm_backend') and self.config.vlm_backend:
                cmd.extend(["-b", self.config.vlm_backend])
            # source
            if hasattr(self.config, 'source') and self.config.source:
                cmd.extend(["--source", self.config.source])
            # language - 使用 -l 参数
            if self.config.language:
                cmd.extend(["-l", self.config.language])

            # 记录命令，使用 shlex.join 来正确显示包含空格的参数
            logger.info(f"Running MinerU CLI: {shlex.join(cmd)}")

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
            
            logger.info(f"Found {len(markdown_files)} markdown files, {len(json_files)} JSON files, {len(image_files)} image files")
            
            # Read markdown content
            markdown_content = ""
            if markdown_files:
                with open(markdown_files[0], 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                logger.info(f"Markdown content length: {len(markdown_content)} characters")
                logger.debug(f"Markdown preview: {markdown_content[:500]}...")
            else:
                logger.warning("No markdown files found in MinerU output")
            
            # Read JSON data
            json_data = {}
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                logger.info(f"JSON data keys: {list(json_data.keys())}")
            else:
                logger.warning("No JSON files found in MinerU output")
            
            # Extract text content from markdown (remove markdown formatting)
            text_content = self._extract_plain_text(markdown_content)
            logger.info(f"Extracted plain text length: {len(text_content)} characters")
            logger.debug(f"Plain text preview: {text_content[:500]}...")
            
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
    
    def _generate_cache_key(self, pdf_path: Path) -> str:
        """
        Generate a unique cache key for a PDF file based on:
        - File path
        - File size
        - Modification time
        - MinerU configuration
        """
        # Get file stats
        stat = pdf_path.stat()
        file_info = f"{pdf_path}:{stat.st_size}:{stat.st_mtime}"
        
        # Include relevant config parameters that affect output
        config_info = f"{self.config.language}:{self.config.vlm_backend}:{self.config.source}"
        
        # Create hash
        combined = f"{file_info}:{config_info}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given cache key"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[DocumentContent]:
        """Load cached DocumentContent if available and valid"""
        try:
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                logger.debug(f"No cache found for key: {cache_key}")
                return None
            
            # Check cache age (configurable expiration)
            cache_age_days = (datetime.now().timestamp() - cache_path.stat().st_mtime) / (24 * 3600)
            if cache_age_days > self.config.cache_expire_days:
                logger.debug(f"Cache expired for key: {cache_key} (age: {cache_age_days:.1f} days, limit: {self.config.cache_expire_days})")
                cache_path.unlink()  # Remove expired cache
                return None
            
            # Load cached content
            with open(cache_path, 'rb') as f:
                cached_content = pickle.load(f)
            
            logger.debug(f"Successfully loaded cached content for key: {cache_key}")
            return cached_content
            
        except Exception as e:
            logger.warning(f"Failed to load cache for key {cache_key}: {str(e)}")
            # If cache loading fails, remove the corrupted cache file
            try:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
            except:
                pass
            return None
    
    def _save_to_cache(self, cache_key: str, content: DocumentContent):
        """Save DocumentContent to cache"""
        try:
            cache_path = self._get_cache_path(cache_key)
            
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(content, f)
            
            logger.debug(f"Successfully cached content for key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache for key {cache_key}: {str(e)}")
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached results
        
        Args:
            older_than_days: If specified, only clear cache older than this many days
        """
        try:
            if not self.cache_dir.exists():
                return
            
            cleared_count = 0
            current_time = datetime.now().timestamp()
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                should_delete = False
                
                if older_than_days is None:
                    should_delete = True
                else:
                    file_age_days = (current_time - cache_file.stat().st_mtime) / (24 * 3600)
                    if file_age_days > older_than_days:
                        should_delete = True
                
                if should_delete:
                    cache_file.unlink()
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} cache files")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache"""
        try:
            if not self.cache_dir.exists():
                return {
                    "cache_dir": str(self.cache_dir),
                    "exists": False,
                    "file_count": 0,
                    "total_size_mb": 0
                }
            
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_dir": str(self.cache_dir),
                "exists": True,
                "file_count": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "files": [
                    {
                        "name": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in cache_files
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {"error": str(e)}