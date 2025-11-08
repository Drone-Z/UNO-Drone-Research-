"""
Reserved message parsing strategy.
"""

from .base import ParsingStrategy
from parse.ads_stan.messages.direct_remote_id import DirectRemoteIdMessage


class ReservedParsingStrategy(ParsingStrategy):
    """Parsing strategy for Reserved messages (type 0x2)."""
    
    def parse(self, payload: bytes) -> DirectRemoteIdMessage:
        """
        Parse Reserved message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            DirectRemoteIdMessage: Parsed Reserved message
        """
        message = DirectRemoteIdMessage(message_type=0x2, version=0x0)
        return message

