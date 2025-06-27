"""
Logging utilities for PDF-to-GraphMD system
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> logging.Logger:
    """
    Setup comprehensive logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("pdf_to_graphmd")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    if not log_file:
        log_file = f"pdf_to_graphmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


class ProgressLogger:
    """Progress tracking logger for long-running operations"""
    
    def __init__(self, logger: logging.Logger, total_items: int, 
                 operation_name: str = "Processing"):
        self.logger = logger
        self.total_items = total_items
        self.operation_name = operation_name
        self.current_item = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1, message: str = ""):
        """Update progress and log status"""
        self.current_item += increment
        
        if self.total_items > 0:
            progress_pct = (self.current_item / self.total_items) * 100
            
            # Calculate ETA
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if self.current_item > 0:
                eta_seconds = (elapsed_time / self.current_item) * (self.total_items - self.current_item)
                eta_str = f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "ETA: unknown"
            
            log_message = f"{self.operation_name}: {self.current_item}/{self.total_items} ({progress_pct:.1f}%) - {eta_str}"
            if message:
                log_message += f" - {message}"
            
            self.logger.info(log_message)
    
    def complete(self, message: str = ""):
        """Mark operation as complete"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        completion_message = f"{self.operation_name} completed in {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
        if message:
            completion_message += f" - {message}"
        
        self.logger.info(completion_message)


def log_exception(logger: logging.Logger, operation: str, exception: Exception):
    """Log exception with context"""
    logger.error(f"Error in {operation}: {str(exception)}", exc_info=True)


def log_performance(logger: logging.Logger, operation: str, 
                   start_time: datetime, end_time: datetime,
                   additional_info: Optional[dict] = None):
    """Log performance metrics"""
    duration = (end_time - start_time).total_seconds()
    
    message = f"Performance - {operation}: {duration:.2f}s"
    
    if additional_info:
        info_str = ", ".join([f"{k}: {v}" for k, v in additional_info.items()])
        message += f" ({info_str})"
    
    logger.info(message)