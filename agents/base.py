# agents/base.py
"""
Base agent class for all DocOps AI agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    name: str = "base"
    description: str = "Base agent"
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Args:
            query: User's input query
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'route', 'success', and agent-specific data
        """
        pass
    
    def __call__(self, query: str, **kwargs) -> Dict[str, Any]:
        """Allow calling agent as function."""
        return self.execute(query, **kwargs)
    
    def log(self, message: str, level: str = "info"):
        """Log a message with the agent's name."""
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{self.name}] {message}")
