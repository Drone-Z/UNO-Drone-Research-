#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
from scapy.all import *

# --------------------------
# User Filters
# This assumes the droen spoofer uses 802.11 Wifi Pakcets
# --------------------------
TARGET_MAC = "AA:BB:CC:DD:EE:FF"  # replace with your spoofed transmitter MAC
TARGET_SSID = None                # set string to filter by SSID
CUSTOM_SIGNATURE = b"\xAA\x11\x55"  # bytes expected in spoofed payload (optional)

# --------------------------
# Setup Monitor Mode
# --------------------------
def enable_monitor_mode(interface):
    print(f"[+] Enabling monitor mode on {interface}")
    os.system(f"sudo ip link set {interface} down")
    os.system(f"sudo iw dev {interface} set type monitor")
    os.system(f"sudo ip link set {interface} up")
    print("[+] Monitor mode enabled")

# --------------------------
# Packet Handler
# --------------------------
def handle_packet(pkt):
    if not pkt.haslayer(Dot11):
        return
    
    # ------ MAC filter ------
    if TARGET_MAC and pkt.addr2 != TARGET_MAC:
        return
    
    # ------ SSID filter ------
    if pkt.haslayer(Dot11Elt):
        ssid = pkt[Dot11Elt].info.decode(errors="ignore")
        if TARGET_SSID and ssid != TARGET_SSID:
            return
    
    # ------ Custom payload filter ------
    raw_payload = bytes(pkt)
    if CUSTOM_SIGNATURE and CUSTOM_SIGNATURE not in raw_payload:
        return

    # Now this is a valid spoofer packet
    print(f"[PACKET] {pkt.addr2} → {pkt.addr1} @ {datetime.now()}")
    
    # Save to JSON
    captured = {
        "timestamp": datetime.now().isoformat(),
        "src": pkt.addr2,
        "dst": pkt.addr1,
        "type": pkt.type,
        "subtype": pkt.subtype,
        "rssi": pkt.dBm_AntSignal if hasattr(pkt, "dBm_AntSignal") else None,
        "raw_hex": raw_payload.hex()
    }

    with open("spoofer_capture.json", "a") as f:
        f.write(json.dumps(captured) + "\n")

# --------------------------
# Main Sniff Loop
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iface", required=True, help="WiFi interface in monitor mode")
    args = parser.parse_args()

    enable_monitor_mode(args.iface)

    print("[+] Starting sniffer… Ctrl+C to stop")
    sniff(iface=args.iface, prn=handle_packet, store=0)

if __name__ == "__main__":
    main()
