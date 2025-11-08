"""
Self ID parsing strategy.
"""

from .base import ParsingStrategy
from parse.ads_stan.messages.self_id import SelfIdMessage


class SelfIdParsingStrategy(ParsingStrategy):
    """Parsing strategy for Self ID messages (type 0x3)."""
    
    def parse(self, payload: bytes) -> SelfIdMessage:
        """
        Parse Self ID message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            SelfIdMessage: Parsed Self ID message
        """
        description_type = payload[0]
        description = payload[1:].decode('ascii', errors='ignore').rstrip('\x00').strip()
        return SelfIdMessage(description_type, description)

