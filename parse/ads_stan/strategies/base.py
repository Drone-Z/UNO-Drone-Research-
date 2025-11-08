"""
Base parsing strategy.
"""

from abc import ABC, abstractmethod
from parse.ads_stan.messages.direct_remote_id import DirectRemoteIdMessage


class ParsingStrategy(ABC):
    """Base class for parsing strategies."""
    
    @abstractmethod
    def parse(self, payload: bytes) -> DirectRemoteIdMessage:
        """
        Parse a message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            DirectRemoteIdMessage: Parsed message
        """
        pass

