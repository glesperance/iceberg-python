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
import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog.rest import RestCatalog


def run_spark_commands(spark: SparkSession, sqls: list[str]) -> None:
    for sql in sqls:
        spark.sql(sql)


@pytest.mark.integration
def test_spark_reads_pyiceberg_v3_metadata(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Verify Spark can read V3 table metadata created by pyiceberg."""
    identifier = "default.pyiceberg_v3_spark_reads"

    run_spark_commands(spark, [f"DROP TABLE IF EXISTS {identifier}"])

    table = session_catalog.create_table(
        identifier,
        schema=pa.schema([("id", pa.int64()), ("data", pa.string())]),
        properties={"format-version": "3"},
    )

    assert table.metadata.format_version == 3

    spark_df = spark.table(identifier)
    assert spark_df.count() == 0
    assert set(spark_df.columns) == {"id", "data"}


@pytest.mark.integration
def test_pyiceberg_reads_spark_v3_table(spark: SparkSession, session_catalog: RestCatalog) -> None:
    """Verify pyiceberg can read a V3 table created by Spark."""
    identifier = "default.spark_v3_pyiceberg_reads"

    run_spark_commands(
        spark,
        [
            f"DROP TABLE IF EXISTS {identifier}",
            f"""
            CREATE TABLE {identifier} (id BIGINT, data STRING)
            USING iceberg
            TBLPROPERTIES ('format-version' = '3')
            """,
            f"INSERT INTO {identifier} VALUES (1, 'a'), (2, 'b'), (3, 'c')",
        ],
    )

    table = session_catalog.load_table(identifier)
    assert table.metadata.format_version == 3

    result = table.scan().to_arrow()
    assert len(result) == 3
    assert set(result["id"].to_pylist()) == {1, 2, 3}
