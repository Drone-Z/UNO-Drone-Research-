"""
Self ID message type (0x3).
"""

from dataclasses import dataclass
from .direct_remote_id import DirectRemoteIdMessage


@dataclass
class SelfIdMessage(DirectRemoteIdMessage):
    """
    Self ID message containing description information.
    """
    # Description type
    ## 0: Text description
    ## 1-200: Reserved
    ## 201-255: Available for private use
    description_type: int

    # Description
    description: str
    
    def __post_init__(self):
        """Initialize the parent class after dataclass initialization."""
        super().__init__(message_type=0x3, version=0x0)

