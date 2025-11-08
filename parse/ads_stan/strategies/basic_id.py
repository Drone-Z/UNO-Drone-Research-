"""
Basic ID parsing strategy.
"""

from .base import ParsingStrategy
from parse.ads_stan.messages.basic_id import BasicIdMessage


class BasicIdParsingStrategy(ParsingStrategy):
    """Parsing strategy for Basic ID messages (type 0x0)."""
    
    def parse(self, payload: bytes) -> BasicIdMessage:
        """
        Parse Basic ID message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            BasicIdMessage: Parsed Basic ID message
        """
        id_type = payload[0] >> 4  # first four bits
        ua_type = payload[0] & 0x0F  # last four bits
        id_number = payload[1:].decode('ascii', errors='ignore').rstrip('\x00').strip()
        return BasicIdMessage(id_type, ua_type, id_number)

