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
"""Integration tests for V3 metadata interoperability with Spark.

These tests verify that V3 table metadata written by pyiceberg can be read
by Spark/Java, and vice versa.
"""

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.table.metadata import TableMetadataV3


def run_spark_commands(spark: SparkSession, sqls: list[str]) -> None:
    for sql in sqls:
        spark.sql(sql)


# ============================================================================
# Test: pyiceberg writes V3 metadata, Spark reads it
# ============================================================================


@pytest.mark.integration
def test_pyiceberg_v3_metadata_spark_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Create V3 table with pyiceberg, verify Spark can read metadata and data."""
    identifier = "default.pyiceberg_v3_metadata_test"

    # Clean up
    run_spark_commands(spark, [f"DROP TABLE IF EXISTS {identifier}"])

    # Create V3 table with pyiceberg
    arrow_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "carol", "dave", "eve"],
            "value": [100, 200, 300, 400, 500],
        }
    )

    table = session_catalog.create_table(
        identifier,
        schema=arrow_table.schema,
        properties={"format-version": "3"},
    )

    # Verify pyiceberg sees V3
    assert table.metadata.format_version == 3
    assert isinstance(table.metadata, TableMetadataV3)

    # Write data
    table.append(arrow_table)

    # Spark should be able to read the table
    spark_df = spark.table(identifier)
    assert spark_df.count() == 5

    # Verify Spark sees correct schema
    spark_schema = spark_df.schema
    assert len(spark_schema.fields) == 3
    assert spark_schema["id"] is not None
    assert spark_schema["name"] is not None
    assert spark_schema["value"] is not None

    # Verify Spark can query the data
    result = spark_df.filter("id > 2").collect()
    assert len(result) == 3
    assert {row.id for row in result} == {3, 4, 5}


@pytest.mark.integration
def test_pyiceberg_v3_metadata_properties(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Verify V3 table properties are visible to Spark."""
    identifier = "default.pyiceberg_v3_props_test"

    run_spark_commands(spark, [f"DROP TABLE IF EXISTS {identifier}"])

    # Create V3 table with custom properties
    arrow_table = pa.table({"id": [1, 2, 3]})

    table = session_catalog.create_table(
        identifier,
        schema=arrow_table.schema,
        properties={
            "format-version": "3",
            "custom.property": "custom-value",
        },
    )

    table.append(arrow_table)

    # Spark should see the table
    spark_df = spark.table(identifier)
    assert spark_df.count() == 3

    # Verify format version via Spark
    spark_props = spark.sql(f"SHOW TBLPROPERTIES {identifier}").collect()
    props_dict = {row.key: row.value for row in spark_props}

    # Spark should see format-version as 3
    # Note: Property name may vary by Spark version
    assert "format-version" in props_dict or "format_version" in props_dict


# ============================================================================
# Test: Spark writes V3 metadata, pyiceberg reads it
# ============================================================================


@pytest.mark.integration
def test_spark_v3_metadata_pyiceberg_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Create V3 table with Spark, verify pyiceberg can read metadata and data."""
    identifier = "default.spark_v3_metadata_test"

    # Create V3 table with Spark
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (
                id BIGINT,
                name STRING,
                value INT
            )
            USING iceberg
            TBLPROPERTIES ('format-version' = '3')
            """,
            f"""
            INSERT INTO {identifier} VALUES
                (1, 'alice', 100),
                (2, 'bob', 200),
                (3, 'carol', 300)
            """,
        ],
    )

    # Load with pyiceberg
    table = session_catalog.load_table(identifier)

    # Verify pyiceberg sees V3 metadata
    assert table.metadata.format_version == 3
    assert isinstance(table.metadata, TableMetadataV3)

    # Verify schema
    schema = table.metadata.schema()
    assert len(schema.fields) == 3
    field_names = {f.name for f in schema.fields}
    assert field_names == {"id", "name", "value"}

    # Verify data can be read
    result = table.scan().to_arrow()
    assert len(result) == 3
    assert set(result["id"].to_pylist()) == {1, 2, 3}


@pytest.mark.integration
def test_spark_v3_with_snapshots_pyiceberg_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Verify pyiceberg can read V3 table with multiple snapshots from Spark."""
    identifier = "default.spark_v3_snapshots_test"

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (id BIGINT, data STRING)
            USING iceberg
            TBLPROPERTIES ('format-version' = '3')
            """,
            f"INSERT INTO {identifier} VALUES (1, 'first')",
            f"INSERT INTO {identifier} VALUES (2, 'second')",
            f"INSERT INTO {identifier} VALUES (3, 'third')",
        ],
    )

    # Load with pyiceberg
    table = session_catalog.load_table(identifier)

    # Verify V3
    assert table.metadata.format_version == 3

    # Should have multiple snapshots
    snapshots = list(table.metadata.snapshots)
    assert len(snapshots) >= 3

    # Current data should have all rows
    result = table.scan().to_arrow()
    assert len(result) == 3


# ============================================================================
# Test: V3 metadata upgrade interop
# ============================================================================


@pytest.mark.integration
def test_pyiceberg_upgrade_v2_to_v3_spark_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Upgrade V2 table to V3 with pyiceberg, verify Spark can still read."""
    identifier = "default.upgrade_v2_v3_interop_test"

    # Create V2 table with Spark
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (id BIGINT, name STRING)
            USING iceberg
            TBLPROPERTIES ('format-version' = '2')
            """,
            f"INSERT INTO {identifier} VALUES (1, 'before'), (2, 'upgrade')",
        ],
    )

    # Verify V2
    table = session_catalog.load_table(identifier)
    assert table.metadata.format_version == 2

    # Upgrade to V3 with pyiceberg
    with table.transaction() as tx:
        tx.upgrade_table_version(3)

    # Reload and verify V3
    table = session_catalog.load_table(identifier)
    assert table.metadata.format_version == 3
    assert isinstance(table.metadata, TableMetadataV3)

    # Spark should still be able to read
    spark_df = spark.table(identifier)
    assert spark_df.count() == 2

    # Spark should be able to write to upgraded table
    run_spark_commands(spark, [f"INSERT INTO {identifier} VALUES (3, 'after')"])

    # Both should see new data
    spark_count = spark.table(identifier).count()
    assert spark_count == 3

    table = session_catalog.load_table(identifier)
    pyiceberg_result = table.scan().to_arrow()
    assert len(pyiceberg_result) == 3


@pytest.mark.integration
def test_spark_upgrade_v2_to_v3_pyiceberg_reads(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Upgrade V2 table to V3 with Spark, verify pyiceberg can read."""
    identifier = "default.spark_upgrade_v2_v3_test"

    # Create V2 table and upgrade with Spark
    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (id BIGINT)
            USING iceberg
            TBLPROPERTIES ('format-version' = '2')
            """,
            f"INSERT INTO {identifier} VALUES (1), (2)",
            # Upgrade to V3
            f"ALTER TABLE {identifier} SET TBLPROPERTIES ('format-version' = '3')",
            f"INSERT INTO {identifier} VALUES (3)",
        ],
    )

    # Load with pyiceberg
    table = session_catalog.load_table(identifier)

    # Should be V3
    assert table.metadata.format_version == 3
    assert isinstance(table.metadata, TableMetadataV3)

    # Should read all data
    result = table.scan().to_arrow()
    assert len(result) == 3
    assert set(result["id"].to_pylist()) == {1, 2, 3}

