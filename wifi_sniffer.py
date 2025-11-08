"""
WiFi Sniffer Module

Handles WiFi interface monitor mode switching and packet capture.
"""

import os
import subprocess
import logging
from typing import Callable, Optional
from scapy.layers.dot11 import Dot11Elt, Dot11EltVendorSpecific
from scapy.sendrecv import AsyncSniffer
from scapy.packet import Packet

# Setup logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def switch_dev_mode(device: str, mode: str) -> bool:
    """
    Changes modes of a device/interface.
    
    Args:
        device: Device/interface that should be changed
        mode: Interface mode, either "monitor" or "managed"
    
    Returns:
        bool: True if change has succeeded, False otherwise
    """
    if mode not in ("monitor", "managed"):
        raise ValueError(f"Only modes 'monitor' and 'managed' are supported, not '{mode}'")
    
    try:
        # Bring interface down
        subprocess.run(
            ["ip", "link", "set", device, "down"],
            check=True,
            capture_output=True,
            timeout=5
        )
        
        # Set mode
        if mode == "monitor":
            subprocess.run(
                ["iwconfig", device, "mode", mode],
                check=True,
                capture_output=True,
                timeout=5
            )
        else:  # managed
            subprocess.run(
                ["iw", "dev", device, "set", "type", mode],
                check=True,
                capture_output=True,
                timeout=5
            )
        
        # Bring interface up
        subprocess.run(
            ["ip", "link", "set", device, "up"],
            check=True,
            capture_output=True,
            timeout=5
        )
        
        return True
    except subprocess.CalledProcessError as e:
        LOG.error(f"Failed to switch {device} to {mode} mode: {e}")
        return False
    except subprocess.TimeoutExpired:
        LOG.error(f"Timeout while switching {device} to {mode} mode")
        return False
    except Exception as e:
        LOG.error(f"Unexpected error switching {device} to {mode} mode: {e}")
        return False


def is_monitor_mode(interface: str) -> bool:
    """
    Check if interface is in monitor mode.
    
    Args:
        interface: Interface name
    
    Returns:
        bool: True if in monitor mode, False otherwise
    """
    try:
        result = subprocess.run(
            ["iw", "dev", interface, "info"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return "type monitor" in result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


class WiFiSniffer:
    """
    Sniffs Wi-Fi interfaces and forwards packets to handlers.
    """
    
    def __init__(self, interface: str, on_packet_received: Callable[[Packet], None],
                 verbose: bool = False):
        """
        Initialize WiFi sniffer.
        
        Args:
            interface: The Wi-Fi device/interface to sniff on
            on_packet_received: Callback function to process received packets
            verbose: Enable verbose logging
        """
        self.interface = interface
        self.on_packet_received = on_packet_received
        self.verbose = verbose
        self.sniffer: Optional[AsyncSniffer] = None
        self.original_mode = None
        
        # Determine original mode
        if is_monitor_mode(interface):
            self.original_mode = "monitor"
        else:
            self.original_mode = "managed"
    
    def start(self) -> bool:
        """
        Sets the interface into monitoring mode and starts sniffing for packets on it.
        
        Returns:
            bool: True when the sniffing has succeeded, False otherwise
        """
        LOG.info(f"Setting interface '{self.interface}' into monitor mode...")
        
        # Switch to monitor mode
        success = switch_dev_mode(self.interface, "monitor")
        
        if success:
            LOG.info(f"Starting sniffer on interface '{self.interface}'...")
            
            # Create and start sniffer
            self.sniffer = AsyncSniffer(
                iface=self.interface,
                prn=self.on_packet_received,
                store=False
            )
            
            self.sniffer.start()
            LOG.info(f"Sniffer on interface '{self.interface}' started")
        else:
            LOG.error(f"Failed to set interface '{self.interface}' into monitor mode")
        
        return success
    
    def stop(self) -> None:
        """
        Stop all sniffing efforts on that interface and restore original mode.
        """
        if self.sniffer:
            LOG.info(f"Stopping sniffer on interface '{self.interface}'...")
            self.sniffer.stop()
            self.sniffer = None
            LOG.info(f"Sniffer on interface '{self.interface}' stopped")
        
        # Restore original mode
        if self.original_mode == "managed":
            LOG.info(f"Restoring interface '{self.interface}' to managed mode...")
            switch_dev_mode(self.interface, "managed")
            LOG.info(f"Interface '{self.interface}' restored to managed mode")

