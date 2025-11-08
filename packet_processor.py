"""
Packet Processor Module

Filters and processes WiFi packets to extract OpenDrone ID messages.
"""

import logging
from typing import Optional
from scapy.layers.dot11 import Dot11EltVendorSpecific
from scapy.packet import Packet

from parse.parser import Parser
from parse.ads_stan.parser import DirectRemoteIdMessageParser

# Setup logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class PacketProcessor:
    """
    Processes WiFi packets to extract and parse OpenDrone ID messages.
    """
    
    # ADS-STAN supported OUIs
    SUPPORTED_OUIS = ["FA:0B:BC", "50:6F:9A", "90:3A:E6"]
    
    def __init__(self, verbose: bool = False):
        """
        Initialize packet processor.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            LOG.setLevel(logging.DEBUG)
    
    def _get_vendor_specific(self, packet: Packet) -> Optional[Dot11EltVendorSpecific]:
        """
        Get the vendor specific layer from the packet.
        
        Args:
            packet: Wi-Fi frame
            
        Returns:
            Dot11EltVendorSpecific: Vendor specific layer, or None if not found
        """
        if packet.haslayer(Dot11EltVendorSpecific):
            return packet.getlayer(Dot11EltVendorSpecific)
        else:
            return None
    
    def _is_ads_stan_oui(self, oui: str) -> bool:
        """
        Check if OUI is supported for ADS-STAN (OpenDrone ID).
        
        Args:
            oui: Organizationally Unique Identifier (format: "AA:BB:CC")
            
        Returns:
            bool: True if OUI is supported, False otherwise
        """
        return oui in self.SUPPORTED_OUIS
    
    def process_packet(self, packet: Packet) -> Optional[object]:
        """
        Process a WiFi packet to extract and parse OpenDrone ID messages.
        
        Method to filter Wi-Fi frames. Only frames containing a vendor specific element
        will not be filtered out directly. After the first filter a second one is applied
        which checks if an OUI of the vendor specific elements belongs to ADS-STAN format.
        If not, it will be dismissed and the next Wi-Fi frame passes through the same
        filter logic.
        
        Args:
            packet: Wi-Fi frame
            
        Returns:
            ParsedMessage if packet contains valid OpenDrone ID message, None otherwise
        """
        vendor_spec: Optional[Dot11EltVendorSpecific] = self._get_vendor_specific(packet)
        
        # Check all vendor-specific elements in the packet
        while vendor_spec:
            # Extract OUI and convert to hex format
            layer_oui = Parser.dec2hex(vendor_spec.oui)
            
            # Check if the packet was sent by a drone (ADS-STAN)
            if self._is_ads_stan_oui(layer_oui):
                # Parse the drone packet
                try:
                    parsed_message = DirectRemoteIdMessageParser.from_wifi(vendor_spec.info, layer_oui)
                    if parsed_message:
                        if self.verbose:
                            LOG.debug(f"Parsed message: {parsed_message}")
                        return parsed_message
                except Exception as e:
                    if self.verbose:
                        LOG.debug(f"Error parsing packet: {e}")
                break
            else:
                # Check next vendor-specific element
                vendor_spec = vendor_spec.payload.getlayer(Dot11EltVendorSpecific)
        
        return None

