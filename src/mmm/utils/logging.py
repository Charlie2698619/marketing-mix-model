"""
Logging utilities for the MMM project.

Provides structured JSON logging with run_id tracking and sensitive data masking.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, mask_keys: Optional[List[str]] = None):
        super().__init__()
        self.mask_keys = mask_keys or ['api_key', 'secret', 'password', 'token']
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'run_id'):
            log_entry['run_id'] = record.run_id
        
        if hasattr(record, 'step'):
            log_entry['step'] = record.step
        
        if hasattr(record, 'duration'):
            log_entry['duration_seconds'] = record.duration
        
        if hasattr(record, 'memory_mb'):
            log_entry['memory_mb'] = record.memory_mb
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Mask sensitive keys
        if hasattr(record, 'extra_data'):
            log_entry['data'] = self._mask_sensitive_data(record.extra_data)
        
        return json.dumps(log_entry, default=str)
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in log entries."""
        if not isinstance(data, dict):
            return data
        
        masked_data = {}
        for key, value in data.items():
            if any(sensitive_key in key.lower() for sensitive_key in self.mask_keys):
                masked_data[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_data(value)
            else:
                masked_data[key] = value
        
        return masked_data


def setup_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: Optional[str] = None,
    mask_keys: Optional[List[str]] = None
) -> None:
    """
    Setup structured logging for the MMM project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format ('json' or 'text')
        log_file: Optional log file path
        mask_keys: List of keys to mask in log output
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Set formatter
    if format.lower() == "json":
        formatter = JSONFormatter(mask_keys=mask_keys)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


class StepLogger:
    """Context manager for logging step execution with timing and memory tracking."""
    
    def __init__(self, step_name: str, run_id: Optional[str] = None):
        self.step_name = step_name
        self.run_id = run_id
        self.logger = logging.getLogger(f"mmm.{step_name}")
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        """Start step logging."""
        import time
        import psutil
        
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create log record with extra fields
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Starting step: {self.step_name}",
            args=(),
            exc_info=None
        )
        
        if self.run_id:
            record.run_id = self.run_id
        record.step = self.step_name
        record.memory_mb = self.start_memory
        
        self.logger.handle(record)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End step logging."""
        import time
        import psutil
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        duration = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        # Determine log level based on success/failure
        if exc_type is None:
            level = logging.INFO
            message = f"Completed step: {self.step_name}"
        else:
            level = logging.ERROR
            message = f"Failed step: {self.step_name} - {exc_val}"
        
        # Create log record with extra fields
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=(exc_type, exc_val, exc_tb) if exc_type else None
        )
        
        if self.run_id:
            record.run_id = self.run_id
        record.step = self.step_name
        record.duration = duration
        record.memory_mb = end_memory
        record.extra_data = {
            'memory_delta_mb': memory_delta,
            'duration_seconds': duration
        }
        
        self.logger.handle(record)


def log_metrics(metrics: Dict[str, Any], run_id: Optional[str] = None, step: Optional[str] = None):
    """Log metrics with structured format."""
    logger = logging.getLogger("mmm.metrics")
    
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Model metrics",
        args=(),
        exc_info=None
    )
    
    if run_id:
        record.run_id = run_id
    if step:
        record.step = step
    
    record.extra_data = metrics
    logger.handle(record)


def log_data_quality(checks: Dict[str, Any], run_id: Optional[str] = None):
    """Log data quality check results."""
    logger = logging.getLogger("mmm.data_quality")
    
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Data quality checks",
        args=(),
        exc_info=None
    )
    
    if run_id:
        record.run_id = run_id
    
    record.extra_data = checks
    logger.handle(record)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_step_logger(step_name: str, run_id: Optional[str] = None) -> logging.Logger:
    """Get a logger for a specific step with enhanced context.
    
    Args:
        step_name: Name of the step
        run_id: Optional run identifier
        
    Returns:
        Logger instance with step context
    """
    logger = logging.getLogger(f"mmm.{step_name}")
    
    # Add step context using extra data
    class StepLoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Add step and run_id to extra if not already present
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            if 'step' not in kwargs['extra']:
                kwargs['extra']['step'] = step_name
            if run_id and 'run_id' not in kwargs['extra']:
                kwargs['extra']['run_id'] = run_id
            return msg, kwargs
    
    return StepLoggerAdapter(logger, {})


if __name__ == "__main__":
    # Example usage
    setup_logging(level="DEBUG", format="json")
    
    logger = logging.getLogger("test")
    
    # Test basic logging
    logger.info("Test message")
    
    # Test step logging
    with StepLogger("test_step", run_id="test_run_123"):
        logger.info("Inside step")
    
    # Test metrics logging
    log_metrics({
        "mape": 0.15,
        "r_squared": 0.85,
        "rhat_max": 1.02
    }, run_id="test_run_123", step="evaluation")
