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
"""Integration tests for Puffin file format interoperability with Spark/Java.

These tests verify that pyiceberg's PuffinWriter produces bytes that are
compatible with the Java Iceberg implementation, and that pyiceberg's
PuffinFile reader can parse DVs written by Spark/Java.

NOTE: These tests focus on the Puffin FILE FORMAT itself, not the full
delete flow. Tests that use table.delete() belong in test_deletion_vectors.py.
"""

from io import BytesIO

import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.manifest import ManifestContent
from pyiceberg.table.puffin import PuffinFile
from pyiceberg.table.puffin_writer import PuffinWriter, serialize_deletion_vector


def run_spark_commands(spark: SparkSession, sqls: list[str]) -> None:
    for sql in sqls:
        spark.sql(sql)


# ============================================================================
# Test: Read Spark-written Puffin DV files with pyiceberg PuffinFile
# ============================================================================


@pytest.mark.integration
def test_read_spark_written_puffin_dv(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Verify pyiceberg's PuffinFile reader can parse Spark-written DVs.

    This test:
    1. Has Spark create a V3 table with data in ONE file (coalesced)
    2. Deletes rows - Spark MUST use DVs since rows share a file
    3. Reads the Puffin file with pyiceberg's PuffinFile class
    4. Verifies the blob metadata and deleted positions are correct

    NOTE: Spark Connect creates one file per row by default. We must coalesce
    to force multiple rows into one file, which requires DVs for partial deletes.
    """
    identifier = "default.spark_puffin_format_test"

    run_spark_commands(spark, [f"DROP TABLE IF EXISTS {identifier}"])

    # Create V3 table
    run_spark_commands(
        spark,
        [
            f"""
            CREATE TABLE {identifier} (id BIGINT)
            USING iceberg
            TBLPROPERTIES (
                'format-version' = '3',
                'write.delete.mode' = 'merge-on-read'
            )
            """,
        ],
    )

    # Insert data coalesced to 1 file - this is KEY for DV creation
    # Spark Connect creates many small files by default, so we must coalesce
    df = spark.range(1, 51)  # 50 rows
    df.coalesce(1).writeTo(identifier).append()

    # Verify we have exactly 1 data file
    files_before = spark.sql(f"SELECT * FROM {identifier}.files").collect()
    assert len(files_before) == 1, f"Expected 1 file, got {len(files_before)}"

    # Delete rows from the middle - Spark MUST use DVs
    run_spark_commands(spark, [f"DELETE FROM {identifier} WHERE id IN (10, 20, 30, 40)"])

    # Load table with pyiceberg
    table = session_catalog.load_table(identifier)

    # Find delete manifests
    current_snapshot = table.current_snapshot()
    assert current_snapshot is not None

    manifests = current_snapshot.manifests(table.io)
    delete_manifests = [m for m in manifests if m.content == ManifestContent.DELETES]

    assert len(delete_manifests) > 0, "Expected delete manifest with DVs"

    # Read delete manifest to find Puffin file
    delete_manifest = delete_manifests[0]
    entries = list(delete_manifest.fetch_manifest_entry(table.io))
    assert len(entries) > 0, "Expected at least one delete file entry"

    delete_entry = entries[0]
    puffin_path = delete_entry.data_file.file_path
    assert puffin_path.endswith(".puffin"), f"Expected Puffin file, got: {puffin_path}"

    # Read the raw Puffin file bytes
    input_file = table.io.new_input(puffin_path)
    with input_file.open() as f:
        puffin_bytes = f.read()

    # Parse with pyiceberg's PuffinFile reader
    puffin = PuffinFile(puffin_bytes)

    # Verify we got deletion vectors
    assert len(puffin.footer.blobs) == 1, "Expected exactly one blob"

    # Verify blob metadata
    blob = puffin.footer.blobs[0]
    assert blob.type == "deletion-vector-v1"
    assert "referenced-data-file" in blob.properties
    assert blob.properties["cardinality"] == "4"  # 4 deleted rows

    # Verify the deleted positions are extracted
    dv_dict = puffin._deletion_vectors
    assert len(dv_dict) == 1, "Expected one data file's deletions"

    # Check positions - should be 4 positions (for IDs 10, 20, 30, 40)
    for data_file, bitmaps in dv_dict.items():
        positions = []
        for bm in bitmaps:
            positions.extend(list(bm))
        assert len(positions) == 4, f"Expected 4 deleted positions, got {len(positions)}"
        # Positions are 0-indexed: ID 10 → pos 9, ID 20 → pos 19, etc.
        assert sorted(positions) == [9, 19, 29, 39], f"Unexpected positions: {positions}"


# ============================================================================
# Test: Read captured DV fixture files
# ============================================================================


@pytest.mark.integration
def test_read_java_format_dv_fixture() -> None:
    """Read a captured Puffin DV file in Java-compatible format.

    This file was created by pyiceberg using the Java-compatible format
    (same format Spark successfully reads in test_pyiceberg_writes_dv_spark_reads).
    Reading it proves our PuffinFile reader handles this format.
    """
    import os

    # Read the fixture file
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "table", "bitmaps", "java_dv_puffin.bin"
    )

    with open(fixture_path, "rb") as f:
        puffin_bytes = f.read()

    # Parse with PuffinFile
    puffin = PuffinFile(puffin_bytes)

    # Verify structure
    assert len(puffin.footer.blobs) == 1
    blob = puffin.footer.blobs[0]

    assert blob.type == "deletion-vector-v1"
    assert "referenced-data-file" in blob.properties
    assert blob.properties["referenced-data-file"] == "s3://bucket/data/file001.parquet"
    assert blob.properties["cardinality"] == "5"

    # Verify positions are extracted
    dv_dict = puffin._deletion_vectors
    assert "s3://bucket/data/file001.parquet" in dv_dict

    bitmaps = dv_dict["s3://bucket/data/file001.parquet"]
    all_positions = set()
    for bm in bitmaps:
        all_positions.update(bm)

    # The fixture has positions 1, 3, 5, 7, 9 (odd numbers)
    assert all_positions == {1, 3, 5, 7, 9}


# ============================================================================
# Test: PuffinWriter/PuffinFile round-trip (no Spark needed)
# ============================================================================


@pytest.mark.integration
def test_puffin_roundtrip_byte_level() -> None:
    """Test that PuffinWriter output can be read by PuffinFile.

    This verifies our writer and reader are consistent with each other,
    independent of Spark. Uses the Java-compatible format.
    """
    from pyroaring import BitMap

    # Create a DV with our writer
    positions = BitMap([1, 5, 10, 100, 1000])

    buf = BytesIO()
    writer = PuffinWriter(buf)
    writer.add_deletion_vector(
        data_file_path="s3://bucket/data/file.parquet",
        positions=positions,
    )
    writer.finish()

    # Read it back with our reader
    buf.seek(0)
    puffin_bytes = buf.read()

    puffin = PuffinFile(puffin_bytes)

    # Verify footer
    assert len(puffin.footer.blobs) == 1
    assert puffin.footer.blobs[0].properties["referenced-data-file"] == "s3://bucket/data/file.parquet"

    # Verify deleted positions
    dv_dict = puffin._deletion_vectors
    assert "s3://bucket/data/file.parquet" in dv_dict

    bitmaps = dv_dict["s3://bucket/data/file.parquet"]
    all_positions = set()
    for bm in bitmaps:
        all_positions.update(bm)

    assert all_positions == {1, 5, 10, 100, 1000}


# ============================================================================
# Test: Verify serialized DV format matches Java spec
# ============================================================================


@pytest.mark.integration
def test_serialize_deletion_vector_format_verification() -> None:
    """Verify the byte-level format of serialize_deletion_vector() matches Java spec.

    Java DV format (Delta-compatible):
    - 4 bytes: length of (magic + bitmap) [big-endian]
    - 4 bytes: magic number 1681511377 [little-endian]
    - variable: RoaringPositionBitmap [little-endian]
    - 4 bytes: CRC-32 checksum [big-endian]
    """
    import zlib

    from pyroaring import BitMap

    positions = BitMap([0, 1, 2])
    serialized = serialize_deletion_vector(positions)

    # Check length prefix (big-endian)
    length = int.from_bytes(serialized[0:4], byteorder="big")
    assert length > 8, f"Length should be > 8, got {length}"

    # Check magic number (little-endian at offset 4)
    magic = int.from_bytes(serialized[4:8], byteorder="little")
    assert magic == 1681511377, f"Magic should be 1681511377, got {magic}"

    # Check CRC (big-endian, at end)
    crc_offset = 4 + length
    stored_crc = int.from_bytes(serialized[crc_offset : crc_offset + 4], byteorder="big")

    # Compute expected CRC (of magic + bitmap, excluding length prefix)
    bitmap_data = serialized[4 : 4 + length]
    computed_crc = zlib.crc32(bitmap_data) & 0xFFFFFFFF

    assert stored_crc == computed_crc, f"CRC mismatch: stored={stored_crc}, computed={computed_crc}"

    # Total size should be: 4 (length) + length + 4 (CRC)
    expected_size = 4 + length + 4
    assert len(serialized) == expected_size, f"Size mismatch: {len(serialized)} != {expected_size}"
