# OpenDrone ID Capture Script

Standalone Python script for capturing and parsing OpenDrone ID packets from WiFi interfaces in monitor mode.

Credit:  This code is modified from https://github.com/cyber-defence-campus/RemoteIDReceiver


## Features

- **Captures OpenDrone ID packets** from WiFi interface in monitor mode
- **Parses ADS-STAN (OpenDrone ID standard) messages** including:
  - Basic ID (0x0)
  - Location Vector (0x1)
  - Self ID (0x3)
  - System Message (0x4)
  - Operator ID (0x5)
  - Message Pack (0xF)
- **Dual output formats**:
  - Human-readable console output
  - JSON format (to stdout or file)
- **Includes raw packet data** in both formats
- **Automatic interface restoration** to managed mode on exit
- **Graceful shutdown** handling (Ctrl+C)

## Requirements

- Python 3.7+
- Root privileges (for monitor mode operations)
- WiFi interface capable of monitor mode
- Required Python packages (see `requirements.txt`)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make the script executable:
```bash
chmod +x capture_opendrone_id.py
```

## Usage

### Basic Usage (Console Output)

```bash
sudo python3 capture_opendrone_id.py -i wlan0
```

### JSON Output to Console

```bash
sudo python3 capture_opendrone_id.py -i wlan0 --json
```

### JSON Output to File

```bash
sudo python3 capture_opendrone_id.py -i wlan0 --json --output packets.json
```

### Verbose Mode

```bash
sudo python3 capture_opendrone_id.py -i wlan0 --verbose
```

## Command-Line Arguments

- `-i, --interface`: WiFi interface to capture on (required, e.g., `wlan0`, `wlan1`)
- `--json`: Output in JSON format (default: human-readable console output)
- `-o, --output`: Output file for JSON format (default: stdout). Ignored if `--json` not specified.
- `-v, --verbose`: Enable verbose logging

## Output Formats

### Console Output

Human-readable format showing:
- Timestamp
- Source MAC address
- Message type name
- All extracted fields (formatted nicely)
- Raw packet data (hex dump)

### JSON Output

Structured JSON with:
- `timestamp`: ISO format timestamp
- `source_mac`: Source MAC address
- `message`: Parsed message object with all fields
- `raw_packet`: 
  - `hex`: Hexadecimal representation
  - `base64`: Base64 encoded
  - `size_bytes`: Packet size in bytes

## Example Output

### Console Output Example

```
================================================================================
Timestamp: 2024-01-15 14:30:25
Source MAC: aa:bb:cc:dd:ee:ff
Message Type: Location Vector (0x1)
Provider: ADS-STAN
--------------------------------------------------------------------------------
  Operational Status: Airborne (2)
  Height Type: AGL (1)
  Position:
    Latitude: 47.3764000°
    Longitude: 8.5400000°
  Altitude:
    Barometric: 10.5 m
    Geodetic: 10.5 m
    Height Above Take-off: 10.5 m
  Movement:
    Track Direction: 190°
    Speed: 5.00 m/s
    Vertical Speed: 7.50 m/s
  Accuracy:
    Horizontal: 2
    Vertical: 3
    Speed: 1
    Barometric Altitude: 4
    Timestamp: 4
  Timestamp: 2024-01-15 14:30:25
--------------------------------------------------------------------------------
Raw Packet Data (hex): aa bb cc dd ee ff 00 11 22 33 44 55 ...
================================================================================
```

### JSON Output Example

```json
{
  "timestamp": "2024-01-15T14:30:25.123456",
  "source_mac": "aa:bb:cc:dd:ee:ff",
  "message": {
    "message_type": 1,
    "message_type_name": "Location Vector",
    "version": 0,
    "provider": "ADS-STAN",
    "operational_status": 2,
    "latitude": 47.3764000,
    "longitude": 8.5400000,
    "speed": 5.00,
    "vertical_speed": 7.50
  },
  "raw_packet": {
    "hex": "aabbccddeeff001122334455...",
    "base64": "qrvM3e7/ABEiM0RV...",
    "size_bytes": 64
  }
}
```

## Supported Message Types

### Basic ID (0x0)
- Identification type
- UA type
- UAS ID

### Location Vector (0x1)
- Operational status
- Position (latitude, longitude)
- Altitude (barometric, geodetic, height above take-off)
- Movement (track direction, speed, vertical speed)
- Accuracy metrics

### Self ID (0x3)
- Description type
- Description text

### System Message (0x4)
- Classification type
- Location source
- Pilot position
- Operating area information
- UA category and class

### Operator ID (0x5)
- Operator ID type
- Operator ID

### Message Pack (0xF)
- Collection of multiple messages

## Notes

- The script automatically switches the interface to monitor mode when starting
- The interface is automatically restored to managed mode when the script exits (Ctrl+C or SIGTERM)
- Requires root privileges for monitor mode operations
- Only ADS-STAN (OpenDrone ID standard) messages are supported (OUIs: FA:0B:BC, 50:6F:9A, 90:3A:E6)
- DJI protocol support is not included in this standalone version

## Troubleshooting

### Interface not found
Make sure the interface name is correct. List available interfaces:
```bash
iwconfig
```

### Permission denied
The script must be run with root privileges:
```bash
sudo python3 capture_opendrone_id.py -i wlan0
```

### Interface cannot be set to monitor mode
Some WiFi adapters may not support monitor mode. Check if your adapter supports it:
```bash
iw phy
```

### No packets captured
- Ensure the interface is in monitor mode
- Check if there are any drones broadcasting OpenDrone ID in range
- Verify the interface is receiving packets: `sudo tcpdump -i wlan0`

## License

See the main project LICENSE file.

