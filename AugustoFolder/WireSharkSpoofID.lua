-- Wireshark Lua Dissectors require Wireshark 2.x+
-- Save as: spoofid.lua inside: ~/.config/wireshark/plugins/

local spoof_proto = Proto("SpoofID", "Drone Spoofer ID Packets")

-- Fields exposed to Wireshark GUI
local f_signature = ProtoField.bytes("spoofid.signature", "Signature Bytes")
local f_payload   = ProtoField.bytes("spoofid.payload", "Payload Data")

spoof_proto.fields = { f_signature, f_payload }

-- Custom signature (must match what your spoofer transmits)
local SIGNATURE = ByteArray.new("AA1155")  -- hex bytes

-- Main dissector
function spoof_proto.dissector(buffer, pinfo, tree)
    local payload = buffer:bytes()

    -- Look for signature inside packet
    if not payload:contains(SIGNATURE) then
        return 0 -- ignore packet
    end

    pinfo.cols.protocol = "SpoofID"
    local subtree = tree:add(spoof_proto, buffer(), "Drone Spoofer Packet")

    subtree:add(f_signature, SIGNATURE)
    subtree:add(f_payload, buffer())
end

-- Bind to ALL WiFi data frames
local wtap_encap = DissectorTable.get("wtap_encap")
wtap_encap:add(wtap.USER0, spoof_proto)
