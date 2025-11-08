"""
System Message parsing strategy.
"""

from .base import ParsingStrategy
import struct
from parse.ads_stan.messages.system_message import SystemMessage


class SystemMessageParsingStrategy(ParsingStrategy):
    """Parsing strategy for System messages (type 0x4)."""
    
    def parse(self, payload: bytes) -> SystemMessage:
        """
        Parse System message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            SystemMessage: Parsed System message
        """
        fmt = '<BiiHBHHBHxxxxx'
        flags, pilot_lat, pilot_lng, area_count, area_radius, area_ceiling, \
        area_floor, ua_cat_class, pilot_geo_alt = struct.unpack(fmt, payload)

        classification_type = (flags & 0b00011100) >> 2  # bits 4..2
        location_source = flags & 0b00000011  # bits 1..0

        pilot_lat /= 10**7
        pilot_lng /= 10**7

        area_radius /= 10
        
        area_ceiling = (area_ceiling * 0.5) - 1000
        area_floor = (area_floor * 0.5) - 1000

        ua_cat = ua_cat_class >> 4  # bits 7..4
        ua_class = ua_cat_class & 0b00001111  # bits 3..0

        pilot_geo_alt = (pilot_geo_alt * 0.5) - 1000
       
        return SystemMessage(
            classification_type, location_source, pilot_lat, pilot_lng,
            area_count, area_radius, area_ceiling, area_floor,
            ua_cat, ua_class, pilot_geo_alt
        )

