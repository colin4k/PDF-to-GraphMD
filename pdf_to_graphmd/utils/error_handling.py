"""
Error handling utilities for PDF-to-GraphMD system
"""
import logging
import traceback
from typing import Optional, Dict, Any, Callable
from functools import wraps
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    PDF_PARSING = "pdf_parsing"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    GRAPH_CONSTRUCTION = "graph_construction"
    OUTPUT_GENERATION = "output_generation"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    API = "api"
    FILE_IO = "file_io"


@dataclass
class SystemError:
    """Structured error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: str
    timestamp: datetime
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    suggested_action: Optional[str] = None


class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_history: list[SystemError] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    suggested_action: Optional[str] = None) -> SystemError:
        """
        Handle and log an error with structured information
        
        Args:
            error: The exception that occurred
            category: Category of the error
            severity: Severity level
            context: Additional context information
            suggested_action: Suggested action to resolve the error
            
        Returns:
            SystemError object with structured information
        """
        
        # Extract traceback information
        tb = traceback.extract_tb(error.__traceback__)
        source_file = None
        line_number = None
        
        if tb:
            last_frame = tb[-1]
            source_file = last_frame.filename
            line_number = last_frame.lineno
        
        # Create structured error
        system_error = SystemError(
            category=category,
            severity=severity,
            message=str(error),
            details=traceback.format_exc(),
            timestamp=datetime.now(),
            source_file=source_file,
            line_number=line_number,
            context=context or {},
            suggested_action=suggested_action
        )
        
        # Log the error
        self._log_error(system_error)
        
        # Store in history
        self.error_history.append(system_error)
        
        # Update counts
        if category not in self.error_counts:
            self.error_counts[category] = 0
        self.error_counts[category] += 1
        
        return system_error
    
    def _log_error(self, error: SystemError):
        """Log error with appropriate level"""
        
        log_message = f"[{error.category.value.upper()}] {error.message}"
        
        if error.context:
            context_str = ", ".join([f"{k}={v}" for k, v in error.context.items()])
            log_message += f" (Context: {context_str})"
        
        if error.suggested_action:
            log_message += f" | Suggested action: {error.suggested_action}"
        
        # Log with appropriate level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:  # LOW
            self.logger.info(log_message)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors"""
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_category": dict(self.error_counts),
            "recent_errors": [
                {
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "timestamp": error.timestamp.isoformat()
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ],
            "critical_errors": [
                error for error in self.error_history 
                if error.severity == ErrorSeverity.CRITICAL
            ]
        }
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors have occurred"""
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.error_history)


def error_handler(category: ErrorCategory, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 suggested_action: Optional[str] = None,
                 reraise: bool = True):
    """
    Decorator for automatic error handling
    
    Args:
        category: Error category
        severity: Error severity level
        suggested_action: Suggested action to resolve the error
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to get error handler from first argument (usually self)
                error_handler_instance = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler_instance = args[0].error_handler
                
                if error_handler_instance:
                    context = {
                        "function": func.__name__,
                        "args": str(args[1:]) if len(args) > 1 else "",
                        "kwargs": str(kwargs)
                    }
                    
                    error_handler_instance.handle_error(
                        e, category, severity, context, suggested_action
                    )
                
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


class RetryHandler:
    """Handle retry logic for transient failures"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, 
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    def retry(self, func: Callable, *args, **kwargs):
        """
        Retry function execution with exponential backoff
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        import time
        
        last_exception = None
        current_delay = self.delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed for {func.__name__}"
                    )
        
        raise last_exception


def safe_execute(func: Callable, default_return=None, 
                error_category: ErrorCategory = ErrorCategory.SYSTEM,
                logger: Optional[logging.Logger] = None):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        default_return: Default return value if function fails
        error_category: Category of potential errors
        logger: Logger instance
        
    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func()
    except Exception as e:
        if logger:
            logger.error(f"Safe execution failed for {func.__name__}: {str(e)}")
        return default_return


def validate_input(value: Any, validator: Callable[[Any], bool], 
                  error_message: str) -> Any:
    """
    Validate input value with custom validator
    
    Args:
        value: Value to validate
        validator: Validation function that returns True if valid
        error_message: Error message if validation fails
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If validation fails
    """
    if not validator(value):
        raise ValueError(error_message)
    return value


class RecoveryStrategies:
    """Common recovery strategies for different error types"""
    
    @staticmethod
    def pdf_parsing_recovery(error: Exception, pdf_path: str) -> Dict[str, str]:
        """Recovery suggestions for PDF parsing errors"""
        suggestions = {
            "FileNotFoundError": f"Ensure PDF file exists at: {pdf_path}",
            "PermissionError": f"Check file permissions for: {pdf_path}",
            "UnicodeDecodeError": "PDF may be corrupted or password-protected",
            "ImportError": "Install required dependencies: pip install magic-pdf",
            "RuntimeError": "Try processing with different MinerU settings"
        }
        
        error_type = type(error).__name__
        return {
            "error_type": error_type,
            "suggestion": suggestions.get(error_type, "Check PDF file integrity and try again")
        }
    
    @staticmethod
    def llm_api_recovery(error: Exception) -> Dict[str, str]:
        """Recovery suggestions for LLM API errors"""
        suggestions = {
            "AuthenticationError": "Check API key configuration",
            "RateLimitError": "Reduce request rate or upgrade API plan",
            "TimeoutError": "Increase timeout setting or retry later",
            "ConnectionError": "Check internet connection and API endpoint",
            "JSONDecodeError": "Enable structured output mode or improve prompts"
        }
        
        error_type = type(error).__name__
        return {
            "error_type": error_type,
            "suggestion": suggestions.get(error_type, "Check API configuration and network connectivity")
        }