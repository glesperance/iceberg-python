# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import math
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pyroaring import BitMap, FrozenBitMap

from pyiceberg.typedef import IcebergBaseModel

if TYPE_CHECKING:
    import pyarrow as pa

# Short for: Puffin Fratercula arctica, version 1
MAGIC_BYTES = b"PFA1"
EMPTY_BITMAP = FrozenBitMap()
MAX_JAVA_SIGNED = int(math.pow(2, 31)) - 1
PROPERTY_REFERENCED_DATA_FILE = "referenced-data-file"

# Java DV format magic number (Delta-compatible)
DV_MAGIC_NUMBER = 1681511377


def _deserialize_roaring_position_bitmap(data: bytes) -> BitMap:
    """Deserialize Java's RoaringPositionBitmap format.

    Format (all little-endian):
    - 8 bytes: number of 32-bit bitmaps
    - For each bitmap:
        - 4 bytes: key (high 32 bits)
        - variable: serialized 32-bit RoaringBitmap
    """
    result = BitMap()
    offset = 0

    # Read count
    num_bitmaps = int.from_bytes(data[offset : offset + 8], byteorder="little")
    offset += 8

    last_key = -1
    for _ in range(num_bitmaps):
        # Read key
        key = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        if key <= last_key:
            raise ValueError("Keys must be sorted in ascending order")
        last_key = key

        # Deserialize 32-bit bitmap
        bitmap_32 = BitMap.deserialize(data[offset:])
        # Skip past the serialized bitmap - need to figure out how many bytes it used
        # Serialize and check length (not ideal but works)
        bitmap_len = len(bitmap_32.serialize())
        offset += bitmap_len

        # Add positions with high bits from key
        high_bits = key << 32
        for pos in bitmap_32:
            result.add(high_bits | pos)

    return result


def _deserialize_java_dv(data: bytes) -> BitMap:
    """Deserialize a deletion vector in Java/Spark format.

    Java format (Delta-compatible):
    - 4 bytes: length of (magic + bitmap) [big-endian]
    - 4 bytes: magic number 1681511377 [little-endian]
    - variable: RoaringPositionBitmap [little-endian]
    - 4 bytes: CRC-32 checksum [big-endian]
    """
    import zlib

    # Read length (big-endian)
    bitmap_data_length = int.from_bytes(data[0:4], byteorder="big")

    # Validate magic number (little-endian, at offset 4)
    magic = int.from_bytes(data[4:8], byteorder="little")
    if magic != DV_MAGIC_NUMBER:
        raise ValueError(f"Invalid DV magic number: {magic}, expected {DV_MAGIC_NUMBER}")

    # Extract bitmap data (after length, includes magic)
    bitmap_data = data[4 : 4 + bitmap_data_length]

    # Verify CRC (checksum of magic + bitmap)
    crc_offset = 4 + bitmap_data_length
    expected_crc = int.from_bytes(data[crc_offset : crc_offset + 4], byteorder="big")
    actual_crc = zlib.crc32(bitmap_data) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"Invalid CRC: {actual_crc}, expected {expected_crc}")

    # Deserialize RoaringPositionBitmap (skip magic, bitmap starts at offset 8)
    bitmap_bytes = data[8 : 4 + bitmap_data_length]
    return _deserialize_roaring_position_bitmap(bitmap_bytes)


def _deserialize_bitmap(pl: bytes) -> list[BitMap]:
    """Deserialize deletion vector blob, auto-detecting format.

    Supports two formats:
    1. Java format: length(4B) + magic(4B) + bitmap + CRC(4B)
    2. Legacy format: num_bitmaps(8B) + [key(4B) + bitmap]*

    Java format is detected by checking for magic number at offset 4.
    """
    # Check if this is Java format by looking for magic number at offset 4
    if len(pl) >= 8:
        potential_magic = int.from_bytes(pl[4:8], byteorder="little")
        if potential_magic == DV_MAGIC_NUMBER:
            # Java format - single bitmap
            return [_deserialize_java_dv(pl)]

    # Legacy format: number_of_bitmaps + [key + bitmap]*
    number_of_bitmaps = int.from_bytes(pl[0:8], byteorder="little")
    pl = pl[8:]

    bitmaps = []
    last_key = -1
    for _ in range(number_of_bitmaps):
        key = int.from_bytes(pl[0:4], byteorder="little")
        if key < 0:
            raise ValueError(f"Invalid unsigned key: {key}")
        if key <= last_key:
            raise ValueError("Keys must be sorted in ascending order")
        if key > MAX_JAVA_SIGNED:
            raise ValueError(f"Key {key} is too large, max {MAX_JAVA_SIGNED} to maintain compatibility with Java impl")
        pl = pl[4:]

        while last_key < key - 1:
            bitmaps.append(EMPTY_BITMAP)
            last_key += 1

        bm = BitMap().deserialize(pl)
        # TODO: Optimize this
        pl = pl[len(bm.serialize()) :]
        bitmaps.append(bm)

        last_key = key

    return bitmaps


class PuffinBlobMetadata(IcebergBaseModel):
    type: Literal["deletion-vector-v1"] = Field()
    fields: list[int] = Field()
    snapshot_id: int = Field(alias="snapshot-id")
    sequence_number: int = Field(alias="sequence-number")
    offset: int = Field()
    length: int = Field()
    compression_codec: str | None = Field(alias="compression-codec", default=None)
    properties: dict[str, str] = Field(default_factory=dict)


class Footer(IcebergBaseModel):
    blobs: list[PuffinBlobMetadata] = Field()
    properties: dict[str, str] = Field(default_factory=dict)


def _bitmaps_to_chunked_array(bitmaps: list[BitMap]) -> "pa.ChunkedArray":
    import pyarrow as pa

    return pa.chunked_array([(key_pos << 32) + pos for pos in bitmap] for key_pos, bitmap in enumerate(bitmaps))


class PuffinFile:
    footer: Footer
    _deletion_vectors: dict[str, list[BitMap]]

    def __init__(self, puffin: bytes) -> None:
        for magic_bytes in [puffin[:4], puffin[-4:]]:
            if magic_bytes != MAGIC_BYTES:
                raise ValueError(f"Incorrect magic bytes, expected {MAGIC_BYTES!r}, got {magic_bytes!r}")

        # One flag is set, the rest should be zero
        # byte 0 (first)
        # - bit 0 (lowest bit): whether FooterPayload is compressed
        # - all other bits are reserved for future use and should be set to 0 on write
        flags = puffin[-8:-4]
        if flags[0] != 0:
            raise ValueError("The Puffin-file has a compressed footer, which is not yet supported")

        # 4 byte integer is always signed, in a two's complement representation, stored little-endian.
        footer_payload_size_int = int.from_bytes(puffin[-12:-8], byteorder="little")

        self.footer = Footer.model_validate_json(puffin[-(footer_payload_size_int + 12) : -12])

        # Use absolute offsets per Puffin spec
        self._deletion_vectors = {
            blob.properties[PROPERTY_REFERENCED_DATA_FILE]: _deserialize_bitmap(puffin[blob.offset : blob.offset + blob.length])
            for blob in self.footer.blobs
        }

    def to_vector(self) -> dict[str, "pa.ChunkedArray"]:
        return {path: _bitmaps_to_chunked_array(bitmaps) for path, bitmaps in self._deletion_vectors.items()}
