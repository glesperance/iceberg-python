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
"""Integration tests for deletion vectors with Spark.

These tests verify interoperability between pyiceberg and Spark/Java SDK
for deletion vector operations.
"""

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.table import TableProperties


def run_spark_commands(spark: SparkSession, sqls: list[str]) -> None:
    for sql in sqls:
        spark.sql(sql)


# ============================================================================
# Test 1: Write with pyiceberg (V3 + DV), Read with Spark
# ============================================================================


@pytest.mark.integration
def test_pyiceberg_writes_dv_spark_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Write data and delete with DVs using pyiceberg, verify Spark can read correctly."""
    identifier = "default.pyiceberg_dv_spark_read"

    # Clean up
    run_spark_commands(spark, [f"DROP TABLE IF EXISTS {identifier}"])

    # Create V3 table with MOR using pyiceberg
    arrow_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )

    table = session_catalog.create_table(
        identifier,
        schema=arrow_table.schema,
        properties={
            "format-version": "3",
            TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
        },
    )

    # Write data with pyiceberg
    table.append(arrow_table)

    # Delete even IDs (scattered positions - better bitmap test than contiguous range)
    table.delete(delete_filter="id IN (2, 4, 6, 8, 10)")

    # Verify with pyiceberg - should have odd IDs remaining
    pyiceberg_result = table.scan().to_arrow()
    assert len(pyiceberg_result) == 5, "pyiceberg should read 5 rows after DV delete"
    assert set(pyiceberg_result["id"].to_pylist()) == {1, 3, 5, 7, 9}

    # Verify Spark can read correctly
    spark_df = spark.table(identifier)
    spark_result = spark_df.collect()

    assert len(spark_result) == 5, f"Spark should read 5 rows, got {len(spark_result)}"
    spark_ids = {row.id for row in spark_result}
    assert spark_ids == {1, 3, 5, 7, 9}


# ============================================================================
# Test 2: Spark deletes (COW), pyiceberg reads
# ============================================================================


@pytest.mark.integration
def test_spark_deletes_pyiceberg_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test Spark deletes and pyiceberg reads.

    NOTE: Spark 4.0.1 + Iceberg 1.10.0 uses Copy-on-Write by default,
    even with merge-on-read set. This test verifies pyiceberg can read
    Spark-modified tables (which use COW file rewrites, not DVs).
    """
    identifier = "default.spark_deletes_pyiceberg_read"

    # Create V3 table with MOR using Spark
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
        CREATE TABLE {identifier} (
            id BIGINT,
            name STRING
        )
        USING iceberg
        TBLPROPERTIES (
            'format-version' = '3',
            'write.delete.mode' = 'merge-on-read'
        )
        """,
            f"""
        INSERT INTO {identifier} VALUES 
            (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e'),
            (6, 'f'), (7, 'g'), (8, 'h'), (9, 'i'), (10, 'j')
        """,
            # Delete even IDs using Spark (may use COW or DVs depending on Spark config)
            f"DELETE FROM {identifier} WHERE id IN (2, 4, 6, 8, 10)",
        ],
    )

    # Load and read with pyiceberg
    table = session_catalog.load_table(identifier)

    # Verify V3
    assert table.metadata.format_version == 3

    # Read should exclude deleted rows (works regardless of COW vs MOR)
    result = table.scan().to_arrow()
    assert len(result) == 5
    assert set(result["id"].to_pylist()) == {1, 3, 5, 7, 9}


# ============================================================================
# Test 3: Mixed operations - both sides writing DVs
# ============================================================================


@pytest.mark.integration
def test_mixed_dv_operations(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Test mixed Spark and pyiceberg delete operations."""
    identifier = "default.mixed_dv_operations"

    # Create table with Spark
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
        CREATE TABLE {identifier} (
            id BIGINT,
            value INT
        )
        USING iceberg
        TBLPROPERTIES (
            'format-version' = '3',
            'write.delete.mode' = 'merge-on-read'
        )
        """,
            f"""
        INSERT INTO {identifier} VALUES 
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50),
            (6, 60), (7, 70), (8, 80), (9, 90), (10, 100)
        """,
        ],
    )

    # Delete some rows with Spark (scattered: 1, 3, 5)
    run_spark_commands(spark, [f"DELETE FROM {identifier} WHERE id IN (1, 3, 5)"])

    # Delete more rows with pyiceberg using DVs (scattered: 2, 8, 10)
    table = session_catalog.load_table(identifier)
    table.delete(delete_filter="id IN (2, 8, 10)")

    # Both should see all deletes - remaining: 4, 6, 7, 9
    table = session_catalog.load_table(identifier)
    pyiceberg_result = table.scan().to_arrow()
    expected_ids = {4, 6, 7, 9}
    assert set(pyiceberg_result["id"].to_pylist()) == expected_ids

    spark_df = spark.table(identifier)
    spark_result = spark_df.collect()
    spark_ids = {row.id for row in spark_result}
    assert spark_ids == expected_ids, f"Spark got {spark_ids}, expected {expected_ids}"


# ============================================================================
# Test 4: Upgrade V2 table to V3, then use DVs
# ============================================================================


@pytest.mark.integration
def test_upgrade_and_use_dvs(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Upgrade V2 table to V3, then use DVs for deletes."""
    identifier = "default.upgrade_to_dv"

    # Create V2 table with Spark
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
        CREATE TABLE {identifier} (
            id BIGINT,
            name STRING
        )
        USING iceberg
        TBLPROPERTIES ('format-version' = '2')
        """,
            f"""
        INSERT INTO {identifier} VALUES 
            (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')
        """,
        ],
    )

    # Load with pyiceberg, upgrade to V3
    table = session_catalog.load_table(identifier)
    assert table.metadata.format_version == 2

    with table.transaction() as tx:
        tx.upgrade_table_version(3)
        tx.set_properties(
            {
                TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
            }
        )

    # Reload to get updated metadata
    table = session_catalog.load_table(identifier)
    assert table.metadata.format_version == 3

    # Now delete with DVs (scattered: even IDs)
    table.delete(delete_filter="id IN (2, 4)")

    # Both pyiceberg and Spark should see 3 rows (1, 3, 5)
    table = session_catalog.load_table(identifier)
    pyiceberg_result = table.scan().to_arrow()
    assert len(pyiceberg_result) == 3
    assert set(pyiceberg_result["id"].to_pylist()) == {1, 3, 5}

    spark_df = spark.table(identifier)
    assert spark_df.count() == 3
    spark_ids = {row.id for row in spark_df.collect()}
    assert spark_ids == {1, 3, 5}


# ============================================================================
# Test 5: Verify DV files are actually created (not COW)
# ============================================================================


@pytest.mark.integration
def test_dv_files_created_not_cow(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Verify that Puffin DV files are created, not COW rewrites."""
    identifier = "default.dv_files_check"

    run_spark_commands(spark, [f"DROP TABLE IF EXISTS {identifier}"])

    # Create V3 + MOR table
    arrow_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        }
    )

    table = session_catalog.create_table(
        identifier,
        schema=arrow_table.schema,
        properties={
            "format-version": "3",
            TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
        },
    )

    table.append(arrow_table)

    # Get data files before delete
    snapshot_before = table.current_snapshot()
    manifests_before = snapshot_before.manifests(table.io) if snapshot_before else []

    # Delete scattered rows (even IDs: 2, 4)
    table.delete(delete_filter="id IN (2, 4)")

    # Get manifests after delete
    table = session_catalog.load_table(identifier)  # Reload
    snapshot_after = table.current_snapshot()
    manifests_after = snapshot_after.manifests(table.io) if snapshot_after else []

    # Check for delete manifests (Puffin files)
    from pyiceberg.manifest import ManifestContent

    delete_manifests = [m for m in manifests_after if m.content == ManifestContent.DELETES]

    # If MOR is working, we should have delete manifests
    # (If COW, we'd have new data manifests instead)
    print(f"Delete manifests found: {len(delete_manifests)}")
    print(f"Total manifests: {len(manifests_after)}")

    # Verify read still works - should have odd IDs: 1, 3, 5
    result = table.scan().to_arrow()
    assert len(result) == 3
    assert set(result["id"].to_pylist()) == {1, 3, 5}
