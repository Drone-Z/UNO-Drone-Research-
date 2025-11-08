#!/usr/bin/env python3
"""
Standalone OpenDrone ID Capture Script

Captures OpenDrone ID packets from WiFi interface in monitor mode,
parses ADS-STAN messages, and outputs both human-readable console
and JSON formats with raw packet data and extracted information.

Requires root privileges for monitor mode operations.
"""

import argparse
import sys
import os
import signal
import json
from datetime import datetime
from typing import Optional

# Check for root privileges
if os.geteuid() != 0:
    print("Error: This script must be run as root (use sudo)")
    print("Monitor mode operations require root privileges.")
    sys.exit(1)

# Import standalone modules
# Add current directory to path for imports
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from wifi_sniffer import WiFiSniffer
from packet_processor import PacketProcessor
from output_formatter import OutputFormatter


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Capture OpenDrone ID packets from WiFi interface in monitor mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo python3 capture_opendrone_id.py -i wlan0
  sudo python3 capture_opendrone_id.py -i wlan0 --json
  sudo python3 capture_opendrone_id.py -i wlan0 --json --output packets.json
        """
    )
    
    parser.add_argument(
        "-i", "--interface",
        required=True,
        help="WiFi interface to capture on (e.g., wlan0, wlan1)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (default: human-readable console output)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for JSON format (default: stdout). Ignored if --json not specified."
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


class OpenDroneIDCapture:
    """
    Main class for capturing and processing OpenDrone ID packets.
    """
    
    def __init__(self, interface: str, json_output: bool = False, 
                 output_file: Optional[str] = None, verbose: bool = False):
        """
        Initialize the capture system.
        
        Args:
            interface: WiFi interface name
            json_output: Whether to output JSON format
            output_file: Optional output file path for JSON
            verbose: Enable verbose logging
        """
        self.interface = interface
        self.json_output = json_output
        self.output_file = output_file
        self.verbose = verbose
        self.running = False
        
        # Initialize components
        self.sniffer = None
        self.processor = PacketProcessor(verbose=verbose)
        self.formatter = OutputFormatter(json_output=json_output, output_file=output_file)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals (SIGINT/SIGTERM).
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print("\n\nReceived shutdown signal. Stopping capture...")
        self.stop()
        sys.exit(0)
    
    def _packet_callback(self, packet):
        """
        Callback function for processing captured packets.
        
        Args:
            packet: Scapy packet object
        """
        try:
            # Process packet and get parsed message
            parsed_message = self.processor.process_packet(packet)
            
            if parsed_message:
                # Format and output the message
                self.formatter.output_message(
                    packet=packet,
                    parsed_message=parsed_message,
                    mac_address=packet.addr2 if hasattr(packet, 'addr2') else None
                )
        except Exception as e:
            if self.verbose:
                print(f"Error processing packet: {e}", file=sys.stderr)
    
    def start(self):
        """
        Start capturing packets.
        """
        print(f"Starting OpenDrone ID capture on interface: {self.interface}")
        print("Press Ctrl+C to stop and restore interface to managed mode\n")
        
        # Create and start sniffer
        self.sniffer = WiFiSniffer(
            interface=self.interface,
            on_packet_received=self._packet_callback,
            verbose=self.verbose
        )
        
        success = self.sniffer.start()
        if not success:
            print(f"Error: Failed to start capture on {self.interface}")
            sys.exit(1)
        
        self.running = True
        
        # Keep running until interrupted
        try:
            import time
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop capturing and restore interface.
        """
        if self.running:
            self.running = False
            if self.sniffer:
                self.sniffer.stop()
            print("\nCapture stopped. Interface restored to managed mode.")


def main():
    """
    Main entry point.
    """
    args = parse_arguments()
    
    # Create capture instance
    capture = OpenDroneIDCapture(
        interface=args.interface,
        json_output=args.json,
        output_file=args.output,
        verbose=args.verbose
    )
    
    # Start capturing
    capture.start()


if __name__ == "__main__":
    main()

