"""
Message Pack parsing strategy.
"""

from .base import ParsingStrategy
from parse.ads_stan.messages.message_pack import MessagePack
from .basic_id import BasicIdParsingStrategy
from .location_vector import LocationVectorParsingStrategy
from .reserved import ReservedParsingStrategy
from .self_id import SelfIdParsingStrategy
from .system_message import SystemMessageParsingStrategy
from .operator_id import OperatorIdParsingStrategy


class MessagePackParsingStrategy(ParsingStrategy):
    """Parsing strategy for Message Pack messages (type 0xF)."""
    
    _strategies = {
        0x0: BasicIdParsingStrategy(),
        0x1: LocationVectorParsingStrategy(),
        0x2: ReservedParsingStrategy(),
        0x3: SelfIdParsingStrategy(),
        0x4: SystemMessageParsingStrategy(),
        0x5: OperatorIdParsingStrategy(),
    }
    
    def parse(self, payload: bytes) -> MessagePack:
        """
        Parse Message Pack payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            MessagePack: Parsed Message Pack containing multiple messages
        """
        n_messages = payload[1]

        messages = []
        for i in range(n_messages):
            message = payload[2 + i * 25: 2 + (i + 1) * 25]
            message_type = message[0] >> 4
            strategy = self._strategies.get(message_type)
            if strategy is None:
                raise ValueError(f"Unknown message type: {message_type}")

            parsed_message = strategy.parse(message[1:])
            messages.append(parsed_message)

        return MessagePack(messages)

