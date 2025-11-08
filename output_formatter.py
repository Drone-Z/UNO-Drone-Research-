"""
Output Formatter Module

Formats parsed OpenDrone ID messages for console and JSON output.
"""

import json
import base64
from datetime import datetime
from typing import Optional
from scapy.packet import Packet

from parse.ads_stan.messages.basic_id import BasicIdMessage
from parse.ads_stan.messages.location_vector import LocationVectorMessage
from parse.ads_stan.messages.self_id import SelfIdMessage
from parse.ads_stan.messages.system_message import SystemMessage
from parse.ads_stan.messages.operator_id import OperatorIdMessage
from parse.ads_stan.messages.message_pack import MessagePack
from parse.ads_stan.messages.direct_remote_id import DirectRemoteIdMessage


class OutputFormatter:
    """
    Formats parsed messages for console and JSON output.
    """
    
    def __init__(self, json_output: bool = False, output_file: Optional[str] = None):
        """
        Initialize output formatter.
        
        Args:
            json_output: Whether to output JSON format
            output_file: Optional output file path for JSON
        """
        self.json_output = json_output
        self.output_file = output_file
        self.output_file_handle = None
        
        if output_file:
            self.output_file_handle = open(output_file, 'a')
    
    def __del__(self):
        """Close output file if opened."""
        if self.output_file_handle:
            self.output_file_handle.close()
    
    def _get_message_type_name(self, message_type: int) -> str:
        """
        Get human-readable message type name.
        
        Args:
            message_type: Message type code
            
        Returns:
            str: Message type name
        """
        type_names = {
            0x0: "Basic ID",
            0x1: "Location Vector",
            0x2: "Reserved",
            0x3: "Self ID",
            0x4: "System",
            0x5: "Operator ID",
            0xF: "Message Pack"
        }
        return type_names.get(message_type, f"Unknown (0x{message_type:X})")
    
    def _format_console_basic_id(self, message: BasicIdMessage) -> str:
        """Format Basic ID message for console output."""
        id_type_names = {0: "None", 1: "Serial Number", 2: "CAA Registration ID"}
        ua_type_names = {
            0: "None/Not Declared", 1: "Aeroplane/Fixed Wing", 2: "Helicopter/Multirotor",
            3: "Gyroplane", 4: "Hybrid Lift", 5: "Ornithopter", 6: "Glider",
            7: "Kite", 8: "Free Balloon", 9: "Captive Balloon", 10: "Airship",
            11: "Free Fall/Parachute", 12: "Rocket", 13: "Tethered Powered Aircraft",
            14: "Ground Obstacle", 15: "Other"
        }
        
        id_type_name = id_type_names.get(message.id_type, f"Unknown ({message.id_type})")
        ua_type_name = ua_type_names.get(message.ua_type, f"Unknown ({message.ua_type})")
        
        return f"""  ID Type: {id_type_name} ({message.id_type})
  UA Type: {ua_type_name} ({message.ua_type})
  UAS ID: {message.uas_id}"""
    
    def _format_console_location_vector(self, message: LocationVectorMessage) -> str:
        """Format Location Vector message for console output."""
        status_names = {0: "Undeclared", 1: "Ground", 2: "Airborne", 3: "Emergency"}
        height_type_names = {0: "Above Take-off", 1: "AGL"}
        
        status_name = status_names.get(message.operational_status, f"Unknown ({message.operational_status})")
        height_type_name = height_type_names.get(message.height_type, f"Unknown ({message.height_type})")
        
        timestamp_str = message.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(message.timestamp, 'strftime') else str(message.timestamp)
        
        return f"""  Operational Status: {status_name} ({message.operational_status})
  Height Type: {height_type_name} ({message.height_type})
  Position:
    Latitude: {message.latitude:.7f}°
    Longitude: {message.longitude:.7f}°
  Altitude:
    Barometric: {message.altitude_barometric:.1f} m
    Geodetic: {message.altitude_geodetic:.1f} m
    Height Above Take-off: {message.height_above_takeoff:.1f} m
  Movement:
    Track Direction: {message.track_direction}°
    Speed: {message.speed:.2f} m/s
    Vertical Speed: {message.vertical_speed:.2f} m/s
  Accuracy:
    Horizontal: {message.accuracy_horizontal}
    Vertical: {message.accuracy_vertical}
    Speed: {message.accuracy_speed}
    Barometric Altitude: {message.accuracy_barometric_altitude}
    Timestamp: {message.accuracy_timestamp}
  Timestamp: {timestamp_str}"""
    
    def _format_console_self_id(self, message: SelfIdMessage) -> str:
        """Format Self ID message for console output."""
        desc_type_names = {0: "Text Description"}
        desc_type_name = desc_type_names.get(message.description_type, f"Private Use ({message.description_type})")
        
        return f"""  Description Type: {desc_type_name} ({message.description_type})
  Description: {message.description}"""
    
    def _format_console_system(self, message: SystemMessage) -> str:
        """Format System message for console output."""
        class_type_names = {0: "Undeclared", 1: "EU"}
        loc_source_names = {0: "Take-Off Location", 1: "Live GNSS", 2: "Fixed Location"}
        ua_cat_names = {0: "Undefined", 1: "Open", 2: "Specific", 3: "Certified"}
        ua_class_names = {0: "Undefined", 1: "Class 0", 2: "Class 1", 3: "Class 2",
                          4: "Class 3", 5: "Class 4", 6: "Class 5", 7: "Class 6"}
        
        class_type_name = class_type_names.get(message.classification_type, f"Reserved ({message.classification_type})")
        loc_source_name = loc_source_names.get(message.location_source, f"Unknown ({message.location_source})")
        ua_cat_name = ua_cat_names.get(message.ua_category, f"Reserved ({message.ua_category})")
        ua_class_name = ua_class_names.get(message.ua_class, f"Reserved ({message.ua_class})")
        
        return f"""  Classification Type: {class_type_name} ({message.classification_type})
  Location Source: {loc_source_name} ({message.location_source})
  Pilot Position:
    Latitude: {message.pilot_latitude:.7f}°
    Longitude: {message.pilot_longitude:.7f}°
    Geodetic Altitude: {message.pilot_geodetic_altitude:.1f} m
  Operating Area:
    Count: {message.area_count}
    Radius: {message.area_radius:.1f} m
    Ceiling: {message.area_ceiling:.1f} m
    Floor: {message.area_floor:.1f} m
  UA Category: {ua_cat_name} ({message.ua_category})
  UA Class: {ua_class_name} ({message.ua_class})"""
    
    def _format_console_operator_id(self, message: OperatorIdMessage) -> str:
        """Format Operator ID message for console output."""
        id_type_names = {0: "Operator ID"}
        id_type_name = id_type_names.get(message.operator_id_type, f"Private Use ({message.operator_id_type})")
        
        return f"""  Operator ID Type: {id_type_name} ({message.operator_id_type})
  Operator ID: {message.operator_id}"""
    
    def _format_console_message_pack(self, message: MessagePack) -> str:
        """Format Message Pack for console output."""
        result = f"  Message Pack containing {len(message.messages)} messages:\n"
        for i, msg in enumerate(message.messages, 1):
            result += f"\n  Message {i}:\n"
            result += self._format_console_message(msg)
        return result
    
    def _format_console_message(self, message: DirectRemoteIdMessage) -> str:
        """Format any message type for console output."""
        if isinstance(message, BasicIdMessage):
            return self._format_console_basic_id(message)
        elif isinstance(message, LocationVectorMessage):
            return self._format_console_location_vector(message)
        elif isinstance(message, SelfIdMessage):
            return self._format_console_self_id(message)
        elif isinstance(message, SystemMessage):
            return self._format_console_system(message)
        elif isinstance(message, OperatorIdMessage):
            return self._format_console_operator_id(message)
        elif isinstance(message, MessagePack):
            return self._format_console_message_pack(message)
        else:
            return f"  Message Type: {message.message_type}, Version: {message.version}"
    
    def _format_console(self, packet: Packet, message: DirectRemoteIdMessage, mac_address: Optional[str]) -> str:
        """
        Format message for console output.
        
        Args:
            packet: Original packet
            message: Parsed message
            mac_address: Source MAC address
            
        Returns:
            str: Formatted console output
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_type_name = self._get_message_type_name(message.message_type)
        
        # Get raw packet data (hex dump)
        raw_data = bytes(packet).hex()
        raw_data_formatted = ' '.join(raw_data[i:i+2] for i in range(0, min(len(raw_data), 64), 2))
        if len(raw_data) > 64:
            raw_data_formatted += " ..."
        
        output = f"""
{'='*80}
Timestamp: {timestamp}
Source MAC: {mac_address or "Unknown"}
Message Type: {message_type_name} (0x{message.message_type:X})
Provider: {message.provider}
{'-'*80}
{self._format_console_message(message)}
{'-'*80}
Raw Packet Data (hex): {raw_data_formatted}
{'='*80}
"""
        return output
    
    def _message_to_dict(self, message: DirectRemoteIdMessage) -> dict:
        """
        Convert message to dictionary for JSON output.
        
        Args:
            message: Parsed message
            
        Returns:
            dict: Message as dictionary
        """
        if isinstance(message, BasicIdMessage):
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider,
                "id_type": message.id_type,
                "ua_type": message.ua_type,
                "uas_id": message.uas_id
            }
        elif isinstance(message, LocationVectorMessage):
            timestamp_str = message.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(message.timestamp, 'strftime') else str(message.timestamp)
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider,
                "operational_status": message.operational_status,
                "is_reserved": message.is_reserved,
                "height_type": message.height_type,
                "track_direction": message.track_direction,
                "speed": message.speed,
                "vertical_speed": message.vertical_speed,
                "latitude": message.latitude,
                "longitude": message.longitude,
                "altitude_barometric": message.altitude_barometric,
                "altitude_geodetic": message.altitude_geodetic,
                "height_above_takeoff": message.height_above_takeoff,
                "timestamp": timestamp_str,
                "accuracy_horizontal": message.accuracy_horizontal,
                "accuracy_vertical": message.accuracy_vertical,
                "accuracy_speed": message.accuracy_speed,
                "accuracy_barometric_altitude": message.accuracy_barometric_altitude,
                "accuracy_timestamp": message.accuracy_timestamp
            }
        elif isinstance(message, SelfIdMessage):
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider,
                "description_type": message.description_type,
                "description": message.description
            }
        elif isinstance(message, SystemMessage):
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider,
                "classification_type": message.classification_type,
                "location_source": message.location_source,
                "pilot_latitude": message.pilot_latitude,
                "pilot_longitude": message.pilot_longitude,
                "pilot_geodetic_altitude": message.pilot_geodetic_altitude,
                "area_count": message.area_count,
                "area_radius": message.area_radius,
                "area_ceiling": message.area_ceiling,
                "area_floor": message.area_floor,
                "ua_category": message.ua_category,
                "ua_class": message.ua_class
            }
        elif isinstance(message, OperatorIdMessage):
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider,
                "operator_id_type": message.operator_id_type,
                "operator_id": message.operator_id
            }
        elif isinstance(message, MessagePack):
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider,
                "messages": [self._message_to_dict(msg) for msg in message.messages]
            }
        else:
            return {
                "message_type": message.message_type,
                "message_type_name": self._get_message_type_name(message.message_type),
                "version": message.version,
                "provider": message.provider
            }
    
    def _format_json(self, packet: Packet, message: DirectRemoteIdMessage, mac_address: Optional[str]) -> dict:
        """
        Format message for JSON output.
        
        Args:
            packet: Original packet
            message: Parsed message
            mac_address: Source MAC address
            
        Returns:
            dict: Formatted JSON data
        """
        timestamp = datetime.now().isoformat()
        raw_data = bytes(packet)
        
        return {
            "timestamp": timestamp,
            "source_mac": mac_address or "Unknown",
            "message": self._message_to_dict(message),
            "raw_packet": {
                "hex": raw_data.hex(),
                "base64": base64.b64encode(raw_data).decode('ascii'),
                "size_bytes": len(raw_data)
            }
        }
    
    def output_message(self, packet: Packet, parsed_message: DirectRemoteIdMessage, mac_address: Optional[str]):
        """
        Output formatted message.
        
        Args:
            packet: Original packet
            parsed_message: Parsed message
            mac_address: Source MAC address
        """
        if self.json_output:
            json_data = self._format_json(packet, parsed_message, mac_address)
            json_str = json.dumps(json_data, indent=2)
            
            if self.output_file_handle:
                self.output_file_handle.write(json_str + "\n")
                self.output_file_handle.flush()
            else:
                print(json_str)
        else:
            console_output = self._format_console(packet, parsed_message, mac_address)
            print(console_output)

