"""
Operator ID parsing strategy.
"""

from .base import ParsingStrategy
from parse.ads_stan.messages.operator_id import OperatorIdMessage


class OperatorIdParsingStrategy(ParsingStrategy):
    """Parsing strategy for Operator ID messages (type 0x5)."""
    
    def parse(self, payload: bytes) -> OperatorIdMessage:
        """
        Parse Operator ID message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            OperatorIdMessage: Parsed Operator ID message
        """
        id_type = payload[0]
        id_number = payload[1:].decode('ascii', errors='ignore').rstrip('\x00').strip()
        return OperatorIdMessage(id_type, id_number)

