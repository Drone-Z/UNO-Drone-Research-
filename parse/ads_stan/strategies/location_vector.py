"""
Location Vector parsing strategy.
"""

from .base import ParsingStrategy
from parse.ads_stan.messages.location_vector import LocationVectorMessage
import struct
import math
from datetime import datetime


class LocationVectorParsingStrategy(ParsingStrategy):
    """Parsing strategy for Location Vector messages (type 0x1)."""
    
    def parse(self, payload: bytes) -> LocationVectorMessage:
        """
        Parse Location Vector message payload.
        
        Args:
            payload: Raw message payload bytes
            
        Returns:
            LocationVectorMessage: Parsed Location Vector message
        """
        status_byte, track_direction, speed, vertical_speed, latitude, longitude, \
        barometric_pressure, geodetic_altitude, height, hv_acc, baro_speed_acc, \
        timestamp, reserved, _ = struct.unpack('BBBbiiHHHBBHBB', payload)

        status = status_byte >> 4  # status is bits 4-7
        is_reserved = (status_byte >> 3) & 0b0001  # reserved is bit 3
        height_type = (status_byte >> 2) & 0b0001  # height_type is bit 2
        direction_sentiment = (status_byte >> 1) & 0b0001  # direction sentiment is bit 1
        speed_multiplier = status_byte & 0b0001  # speed multiplier is bit 0

        track_direction = track_direction + 180 if direction_sentiment else track_direction

        if not speed_multiplier:
            speed *= 0.25
        else:
            speed *= 0.75
            speed += 255 * 0.25
            
        vertical_speed *= 0.5
        latitude /= 10 ** 7
        longitude /= 10 ** 7
        
        barometric_pressure = (barometric_pressure * 0.5) - 1000
        geodetic_altitude = (geodetic_altitude * 0.5) - 1000
        height = (height * 0.5) - 1000
        
        v_acc = (hv_acc >> 4) & 0b00001111  # bits 7..4
        h_acc = hv_acc & 0b00001111  # bits 3..0
        baro_acc = (baro_speed_acc >> 4) & 0b00001111  # bits 7..4
        speed_acc = baro_speed_acc & 0b00001111  # bits 3..0
        timestamp_acc = reserved & 0b00001111  # bits 3..0

        # Parse timestamp
        min_ = math.floor(timestamp / 600)
        sec = round((timestamp - min_ * 600) / 10)
        now = datetime.now()
        tenth_second = now.minute * 600 + now.second * 10
        if timestamp > tenth_second:
            # Timestamp is from previous hour
            if now.hour > 0:
                timestamp_dt = now.replace(hour=now.hour-1, minute=min_, second=sec, microsecond=0)
            else:
                # Previous day
                from datetime import timedelta
                timestamp_dt = (now - timedelta(hours=1)).replace(minute=min_, second=sec, microsecond=0)
        else:
            timestamp_dt = now.replace(minute=min_, second=sec, microsecond=0)
        
        return LocationVectorMessage(
            status, is_reserved, height_type, track_direction, speed, vertical_speed,
            latitude, longitude, barometric_pressure, geodetic_altitude, height,
            timestamp_dt, h_acc, v_acc, speed_acc, baro_acc, timestamp_acc
        )

