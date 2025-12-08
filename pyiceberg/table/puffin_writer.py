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
"""Puffin file writer for deletion vectors.

This module provides write support for Puffin files containing deletion vectors,
complementing the read support in puffin.py.
"""

from __future__ import annotations

import json
import zlib
from io import BytesIO
from typing import BinaryIO

from pyroaring import BitMap

# Puffin file magic bytes: "PFA1"
MAGIC = b"PFA1"

# Deletion vector magic number (little-endian) for Delta compatibility
DV_MAGIC = 1681511377

# Standard blob type for deletion vectors
BLOB_TYPE_DV_V1 = "deletion-vector-v1"

# Property key for referenced data file
PROPERTY_REFERENCED_DATA_FILE = "referenced-data-file"
PROPERTY_CARDINALITY = "cardinality"


def _serialize_roaring_position_bitmap(positions: BitMap) -> bytes:
    """Serialize a BitMap in Java's RoaringPositionBitmap format.
    
    Format (all little-endian):
    - 8 bytes: number of 32-bit bitmaps
    - For each non-empty bitmap:
        - 4 bytes: key (high 32 bits of positions)
        - variable: serialized 32-bit RoaringBitmap
    
    This is compatible with Java's RoaringPositionBitmap.deserialize().
    """
    # Group positions by high 32 bits (key)
    bitmaps_by_key: dict[int, BitMap] = {}
    for pos in positions:
        key = pos >> 32  # High 32 bits
        low = pos & 0xFFFFFFFF  # Low 32 bits
        if key not in bitmaps_by_key:
            bitmaps_by_key[key] = BitMap()
        bitmaps_by_key[key].add(low)
    
    result = bytearray()
    
    # Write count of non-empty bitmaps
    result.extend(len(bitmaps_by_key).to_bytes(8, byteorder="little"))
    
    # Write each bitmap in key order
    for key in sorted(bitmaps_by_key.keys()):
        bitmap = bitmaps_by_key[key]
        bitmap.run_optimize()
        
        # Write key
        result.extend(key.to_bytes(4, byteorder="little"))
        
        # Write serialized bitmap
        result.extend(bitmap.serialize())
    
    return bytes(result)


def serialize_deletion_vector(positions: BitMap | list[int] | set[int]) -> bytes:
    """Serialize a deletion vector in Java/Spark format (Delta-compatible).
    
    Format:
    - 4 bytes: length of (magic + bitmap) [big-endian]
    - 4 bytes: magic number 1681511377 [little-endian]
    - variable: RoaringPositionBitmap [little-endian]
    - 4 bytes: CRC-32 checksum of (magic + bitmap) [big-endian]
    
    This format is compatible with Java Iceberg and Spark.
    
    Args:
        positions: Deleted row positions as a BitMap, list, or set
        
    Returns:
        Serialized deletion vector bytes
    """
    if not isinstance(positions, BitMap):
        positions = BitMap(positions)
    
    # Serialize in Java's RoaringPositionBitmap format
    bitmap_bytes = _serialize_roaring_position_bitmap(positions)
    
    # Build wrapper: magic(4B little) + bitmap_bytes
    magic_and_bitmap = DV_MAGIC.to_bytes(4, byteorder="little") + bitmap_bytes
    bitmap_data_length = len(magic_and_bitmap)
    
    # Assemble: length(4B big) + magic_and_bitmap + CRC(4B big)
    result = bytearray()
    result.extend(bitmap_data_length.to_bytes(4, byteorder="big"))
    result.extend(magic_and_bitmap)
    
    # CRC32 of magic + bitmap (not including length prefix)
    crc = zlib.crc32(magic_and_bitmap) & 0xFFFFFFFF
    result.extend(crc.to_bytes(4, byteorder="big"))
    
    return bytes(result)


class PuffinWriter:
    """Writer for Puffin files containing deletion vectors.
    
    Puffin file format:
    - Magic (4 bytes): "PFA1"
    - Blobs: serialized deletion vectors
    - Magic (4 bytes): "PFA1" (footer start marker)
    - Footer payload (JSON)
    - Footer payload length (4 bytes, little-endian)
    - Flags (4 bytes)
    - Magic (4 bytes): "PFA1"
    
    Example:
        >>> from io import BytesIO
        >>> from pyroaring import BitMap
        >>> 
        >>> buf = BytesIO()
        >>> writer = PuffinWriter(buf)
        >>> writer.add_deletion_vector(
        ...     data_file_path="s3://bucket/data.parquet",
        ...     positions=BitMap([1, 5, 10]),
        ... )
        >>> writer.finish()
        >>> 
        >>> # Read it back
        >>> buf.seek(0)
        >>> from pyiceberg.table.puffin import PuffinFile
        >>> puffin = PuffinFile(buf.read())
    """
    
    def __init__(self, output: BinaryIO, properties: dict[str, str] | None = None):
        """Initialize a Puffin writer.
        
        Args:
            output: Binary file-like object to write to
            properties: Optional file-level properties
        """
        self._output = output
        self._properties = properties or {}
        self._blobs: list[dict] = []
        self._pos = 0
        self._finished = False
        
        # Write header magic
        self._write(MAGIC)
    
    def _write(self, data: bytes) -> int:
        """Write data and track position."""
        written = self._output.write(data)
        self._pos += written
        return written
    
    def add_deletion_vector(
        self,
        data_file_path: str,
        positions: BitMap | list[int] | set[int],
        snapshot_id: int = -1,
        sequence_number: int = -1,
    ) -> None:
        """Add a deletion vector for a data file.
        
        Args:
            data_file_path: Path to the data file these deletes reference
            positions: Row positions to mark as deleted
            snapshot_id: Snapshot ID (-1 for inherited)
            sequence_number: Sequence number (-1 for inherited)
        """
        if self._finished:
            raise RuntimeError("Cannot add blobs after finish()")
        
        if not isinstance(positions, BitMap):
            positions = BitMap(positions)
        
        # Record absolute offset (from start of file, per Puffin spec)
        offset = self._pos
        
        # Serialize and write the deletion vector (Java-compatible format)
        blob_data = serialize_deletion_vector(positions)
        self._write(blob_data)
        
        # Track blob metadata for footer
        self._blobs.append({
            "type": BLOB_TYPE_DV_V1,
            "fields": [],
            "snapshot-id": snapshot_id,
            "sequence-number": sequence_number,
            "offset": offset,
            "length": len(blob_data),
            "properties": {
                PROPERTY_REFERENCED_DATA_FILE: data_file_path,
                PROPERTY_CARDINALITY: str(len(positions)),
            },
        })
    
    def finish(self) -> int:
        """Finish writing and return the file size.
        
        Returns:
            Total file size in bytes
        """
        if self._finished:
            raise RuntimeError("Already finished")
        
        # Write footer start magic
        self._write(MAGIC)
        
        # Build and write footer JSON
        footer = {"blobs": self._blobs}
        if self._properties:
            footer["properties"] = self._properties
        
        footer_json = json.dumps(footer, separators=(",", ":")).encode("utf-8")
        self._write(footer_json)
        footer_length = len(footer_json)
        
        # Write footer length (little-endian)
        self._write(footer_length.to_bytes(4, byteorder="little"))
        
        # Write flags (no compression = all zeros)
        self._write(b"\x00\x00\x00\x00")
        
        # Write final magic
        self._write(MAGIC)
        
        self._finished = True
        return self._pos
    
    @property
    def file_size(self) -> int:
        """Return current file size."""
        return self._pos


def write_deletion_vectors_to_puffin(
    output: BinaryIO,
    deletion_vectors: dict[str, BitMap | list[int] | set[int]],
    snapshot_id: int = -1,
    sequence_number: int = -1,
    properties: dict[str, str] | None = None,
) -> int:
    """Convenience function to write multiple deletion vectors to a Puffin file.
    
    Args:
        output: Binary file-like object to write to
        deletion_vectors: Mapping of data file path to deleted positions
        snapshot_id: Snapshot ID for all blobs (-1 for inherited)
        sequence_number: Sequence number for all blobs (-1 for inherited)
        properties: Optional file-level properties
        
    Returns:
        Total file size in bytes
        
    Example:
        >>> from io import BytesIO
        >>> buf = BytesIO()
        >>> write_deletion_vectors_to_puffin(
        ...     buf,
        ...     {
        ...         "s3://bucket/file1.parquet": [1, 2, 3],
        ...         "s3://bucket/file2.parquet": [10, 20, 30],
        ...     }
        ... )
    """
    writer = PuffinWriter(output, properties)
    
    for data_file_path, positions in deletion_vectors.items():
        writer.add_deletion_vector(
            data_file_path=data_file_path,
            positions=positions,
            snapshot_id=snapshot_id,
            sequence_number=sequence_number,
        )
    
    return writer.finish()

