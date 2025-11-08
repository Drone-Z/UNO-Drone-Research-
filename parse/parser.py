"""
Base Parser for Remote ID messages.
"""

import struct
from abc import ABC
from typing import Optional, List
from scapy.packet import Packet


class ParseRemoteIdError(Exception):
    """
    Raised when the parsing of a Wi-Fi packet to a Remote ID-object could not successfully complete.
    """
    pass


class ParsedMessage(ABC):
    """Represents a parsed Remote ID message from any supported protocol."""
    provider: str

    def __init__(self, provider: str):
        self.provider = provider


class Parser:
    """
    Root Parser for a vendor specific packet.
    """
    header_size = 8
    oui: List[str] = []  # List of supported OUIs

    @staticmethod
    def dec2hex(oui_dec: int) -> str:
        """
        Method to parse the decimal value of the OUI to a readable and formatted hex value of the OUI.
        The format of the OUI is according to IEEE either AB:CD:EF or without colon ABCDEF.
        This method parsed the OUI to the first mentioned format -> AC:CD:EF.

        Args:
            oui_dec: Decimal value of OUI.

        Returns:
            str: Formatted OUI.
        """
        max_ = 16777215
        min_ = 0
        if oui_dec < min_ or oui_dec > max_:
            return "00:00:00"
        oui_raw = hex(oui_dec)[2:].zfill(6)  # [2:0] -> to remove '0x' part of hex value
        return f"{oui_raw[0:2]}:{oui_raw[2:4]}:{oui_raw[4:]}".upper()

    @staticmethod
    def from_wifi(packet: Packet, oui: str) -> Optional[ParsedMessage]:
        """
        Parse a vendor specific element of a Wi-Fi packet.

        Args:
            packet: Wi-Fi packet.
            oui: Vendor OUI.

        Returns:
            Optional[ParsedMessage]: Parsed RemoteId or None if parsing not possible.
        """
        raise NotImplementedError("Subclasses must implement from_wifi method")

