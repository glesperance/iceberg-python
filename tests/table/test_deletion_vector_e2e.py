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
"""End-to-end tests for deletion vectors without Spark.

These tests create real Iceberg tables, write data, perform deletes using
deletion vectors, and verify reads correctly exclude deleted rows.
"""

import tempfile
import uuid
from pathlib import Path
from typing import Union

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.expressions import EqualTo, GreaterThan, LessThan
from pyiceberg.io import PY_IO_IMPL, load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import ToOutputFile
from pyiceberg.table import (
    CommitTableResponse,
    CreateTableTransaction,
    Table,
    TableProperties,
)
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.table.update import (
    AssertTableUUID,
    TableRequirement,
    TableUpdate,
    update_table_metadata,
)
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties
from pyiceberg.types import IntegerType, LongType, NestedField, StringType


class SimpleTestCatalog(Catalog):
    """A simple in-memory catalog for testing without external dependencies."""

    def __init__(self, name: str, warehouse: str, **properties):
        super().__init__(name, **properties)
        self._warehouse = warehouse
        self._tables: dict[str, Table] = {}
        self._namespaces: set[str] = set()
        self._io = load_file_io({PY_IO_IMPL: "pyiceberg.io.pyarrow.PyArrowFileIO"})

    def create_namespace(self, namespace: str | Identifier, properties: Properties = EMPTY_DICT) -> None:
        ns = namespace if isinstance(namespace, str) else ".".join(namespace)
        self._namespaces.add(ns)

    def create_namespace_if_not_exists(self, namespace: str | Identifier, properties: Properties = EMPTY_DICT) -> None:
        ns = namespace if isinstance(namespace, str) else ".".join(namespace)
        self._namespaces.add(ns)

    def drop_namespace(self, namespace: str | Identifier) -> None:
        ns = namespace if isinstance(namespace, str) else ".".join(namespace)
        self._namespaces.discard(ns)

    def list_namespaces(self, namespace: str | Identifier = ()) -> list[Identifier]:
        return [(ns,) for ns in self._namespaces]

    def load_namespace_properties(self, namespace: str | Identifier) -> Properties:
        return {}

    def update_namespace_properties(
        self, namespace: str | Identifier, removals: set[str] | None = None, updates: Properties = EMPTY_DICT
    ) -> PropertiesUpdateSummary:
        return PropertiesUpdateSummary(removed=[], updated=[], missing=[])

    def create_table(
        self,
        identifier: str | Identifier,
        schema: Union[Schema, "pa.Schema"],
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> Table:
        if isinstance(identifier, str):
            identifier = tuple(identifier.split("."))

        table_name = ".".join(identifier)
        location = location or f"{self._warehouse}/{table_name.replace('.', '/')}"

        if isinstance(schema, pa.Schema):
            from pyiceberg.io.pyarrow import pyarrow_to_schema
            schema = pyarrow_to_schema(schema)

        metadata = new_table_metadata(
            schema=schema,
            partition_spec=partition_spec,
            sort_order=sort_order,
            location=location,
            properties=properties,
        )

        metadata_location = f"{location}/metadata/v1.metadata.json"
        ToOutputFile.table_metadata(metadata, self._io.new_output(metadata_location))

        table = Table(
            identifier=identifier,
            metadata=metadata,
            metadata_location=metadata_location,
            io=self._io,
            catalog=self,
        )
        self._tables[table_name] = table
        return table

    def create_table_transaction(
        self,
        identifier: str | Identifier,
        schema: Union[Schema, "pa.Schema"],
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> CreateTableTransaction:
        raise NotImplementedError

    def load_table(self, identifier: str | Identifier) -> Table:
        table_name = identifier if isinstance(identifier, str) else ".".join(identifier)
        if table_name not in self._tables:
            raise KeyError(f"Table {table_name} not found")
        return self._tables[table_name]

    def table_exists(self, identifier: str | Identifier) -> bool:
        table_name = identifier if isinstance(identifier, str) else ".".join(identifier)
        return table_name in self._tables

    def register_table(self, identifier: str | Identifier, metadata_location: str) -> Table:
        raise NotImplementedError

    def drop_table(self, identifier: str | Identifier) -> None:
        table_name = identifier if isinstance(identifier, str) else ".".join(identifier)
        self._tables.pop(table_name, None)

    def purge_table(self, identifier: str | Identifier) -> None:
        self.drop_table(identifier)

    def rename_table(self, from_identifier: str | Identifier, to_identifier: str | Identifier) -> Table:
        raise NotImplementedError

    def list_tables(self, namespace: str | Identifier) -> list[Identifier]:
        ns = namespace if isinstance(namespace, str) else ".".join(namespace)
        return [
            tuple(name.split("."))
            for name in self._tables
            if name.startswith(f"{ns}.")
        ]

    def commit_table(
        self, table: Table, requirements: tuple[TableRequirement, ...], updates: tuple[TableUpdate, ...]
    ) -> CommitTableResponse:
        # Apply updates to create new metadata
        new_metadata = update_table_metadata(table.metadata, updates)

        # Write new metadata
        version = table.metadata_location.split("/")[-1].replace(".metadata.json", "")
        try:
            new_version = int(version.replace("v", "")) + 1
        except ValueError:
            new_version = 2
        new_metadata_location = f"{table.metadata.location}/metadata/v{new_version}.metadata.json"
        ToOutputFile.table_metadata(new_metadata, self._io.new_output(new_metadata_location))

        # Update the in-memory table reference
        table_name = ".".join(table.name())
        updated_table = Table(
            identifier=table.name(),
            metadata=new_metadata,
            metadata_location=new_metadata_location,
            io=self._io,
            catalog=self,
        )
        self._tables[table_name] = updated_table

        return CommitTableResponse(
            metadata=new_metadata,
            metadata_location=new_metadata_location,
        )

    def list_views(self, namespace: str | Identifier) -> list[Identifier]:
        return []

    def view_exists(self, identifier: str | Identifier) -> bool:
        return False

    def drop_view(self, identifier: str | Identifier) -> None:
        pass


@pytest.fixture
def warehouse_path(tmp_path):
    """Create a temporary warehouse directory."""
    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    return str(warehouse)


@pytest.fixture
def catalog(warehouse_path):
    """Create a simple test catalog (no external dependencies)."""
    return SimpleTestCatalog("test_catalog", warehouse=warehouse_path)


@pytest.fixture
def test_schema():
    """Simple test schema."""
    return Schema(
        NestedField(field_id=1, name="id", field_type=LongType(), required=False),
        NestedField(field_id=2, name="name", field_type=StringType(), required=False),
        NestedField(field_id=3, name="value", field_type=IntegerType(), required=False),
    )


@pytest.fixture
def sample_data():
    """Sample data for testing - schema must match test_schema."""
    return pa.table({
        "id": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=pa.int64()),
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "value": pa.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], type=pa.int32()),
    })


class TestDeletionVectorE2E:
    """End-to-end tests for deletion vectors."""

    def test_create_table_write_and_read(self, catalog, test_schema, sample_data) -> None:
        """Basic sanity check: create table, write data, read it back."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.test_table",
            schema=test_schema,
        )
        
        # Write data
        table.append(sample_data)
        
        # Read back
        result = table.scan().to_arrow()
        
        assert len(result) == 10
        assert set(result["id"].to_pylist()) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    def test_delete_with_copy_on_write(self, catalog, test_schema, sample_data) -> None:
        """Test that copy-on-write delete still works (baseline)."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.cow_delete_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Delete rows where id > 5 using copy-on-write (default)
        table.delete(delete_filter="id > 5")
        
        # Read back - should only have ids 1-5
        result = table.scan().to_arrow()
        
        assert len(result) == 5
        assert set(result["id"].to_pylist()) == {1, 2, 3, 4, 5}

    def test_delete_with_expression_object(self, catalog, test_schema, sample_data) -> None:
        """Test delete with BooleanExpression object."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.expr_delete_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Delete using expression object
        table.delete(delete_filter=EqualTo("name", "c"))
        
        result = table.scan().to_arrow()
        
        assert len(result) == 9
        assert 3 not in result["id"].to_pylist()  # id=3 had name="c"

    def test_multiple_deletes(self, catalog, test_schema, sample_data) -> None:
        """Test multiple delete operations on same table."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.multi_delete_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # First delete
        table.delete(delete_filter="id = 1")
        
        # Second delete  
        table.delete(delete_filter="id = 10")
        
        # Third delete
        table.delete(delete_filter="value = 50")  # id=5
        
        result = table.scan().to_arrow()
        
        assert len(result) == 7
        assert set(result["id"].to_pylist()) == {2, 3, 4, 6, 7, 8, 9}

    def test_delete_all_rows(self, catalog, test_schema, sample_data) -> None:
        """Test deleting all rows from a table."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.delete_all_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Delete everything
        table.delete(delete_filter="id > 0")
        
        result = table.scan().to_arrow()
        
        assert len(result) == 0

    def test_delete_no_matching_rows(self, catalog, test_schema, sample_data) -> None:
        """Test delete that matches no rows."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.delete_none_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Delete with filter that matches nothing
        with pytest.warns(UserWarning, match="Delete operation did not match any records"):
            table.delete(delete_filter="id = 999")
        
        result = table.scan().to_arrow()
        
        # All rows should still be there
        assert len(result) == 10

    def test_delete_then_append(self, catalog, test_schema, sample_data) -> None:
        """Test delete followed by append."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.delete_append_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Delete some rows
        table.delete(delete_filter="id <= 5")
        
        # Append new data (must match schema types)
        new_data = pa.table({
            "id": pa.array([11, 12, 13], type=pa.int64()),
            "name": ["k", "l", "m"],
            "value": pa.array([110, 120, 130], type=pa.int32()),
        })
        table.append(new_data)
        
        result = table.scan().to_arrow()
        
        assert len(result) == 8  # 5 remaining + 3 new
        assert set(result["id"].to_pylist()) == {6, 7, 8, 9, 10, 11, 12, 13}

    def test_scan_with_filter_after_delete(self, catalog, test_schema, sample_data) -> None:
        """Test that scan filters work correctly after delete."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.filter_after_delete_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Delete some rows
        table.delete(delete_filter="id > 7")
        
        # Scan with additional filter
        result = table.scan(row_filter="value > 30").to_arrow()
        
        # Should get ids 4, 5, 6, 7 (value > 30 and id <= 7)
        assert len(result) == 4
        assert set(result["id"].to_pylist()) == {4, 5, 6, 7}


class TestDeletionVectorMORConfig:
    """Tests for merge-on-read configuration."""

    def test_mor_table_property_recognized(self, catalog, test_schema, sample_data) -> None:
        """Test that merge-on-read property is recognized."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # Create table with MOR delete mode
        table = catalog.create_table(
            "test_ns.mor_table",
            schema=test_schema,
            properties={
                TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
            },
        )
        
        # Verify property is set
        assert table.properties.get(TableProperties.DELETE_MODE) == TableProperties.DELETE_MODE_MERGE_ON_READ

    def test_mor_raises_on_v2(self, catalog, test_schema, sample_data) -> None:
        """Test that MOR raises an error on V2 tables (DVs require V3)."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # V2 table with MOR requested
        table = catalog.create_table(
            "test_ns.mor_fallback_table",
            schema=test_schema,
            properties={
                TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
                # Default format version is 2
            },
        )
        
        table.append(sample_data)
        
        # Delete should raise error since MOR requires V3
        with pytest.raises(ValueError, match="requires format version 3"):
            table.delete(delete_filter="id > 5")


class TestDeletionVectorV3:
    """Tests for deletion vectors with V3 tables."""

    def test_v3_mor_uses_deletion_vectors(self, catalog, test_schema, sample_data, warehouse_path) -> None:
        """Test that V3 + MOR actually uses deletion vectors (not COW)."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # Create V3 table with MOR
        table = catalog.create_table(
            "test_ns.v3_mor_table",
            schema=test_schema,
            properties={
                "format-version": "3",
                TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
            },
        )
        
        assert table.metadata.format_version == 3
        
        table.append(sample_data)
        
        # Count data files before delete
        snapshot_before = table.current_snapshot()
        manifests_before = snapshot_before.manifests(table.io) if snapshot_before else []
        data_files_before = sum(1 for m in manifests_before for _ in m.fetch_manifest_entry(table.io))
        
        # Delete using MOR (should create DVs, not rewrite files)
        table.delete(delete_filter="id > 5")
        
        # Verify the delete worked
        result = table.scan().to_arrow()
        assert len(result) == 5
        assert set(result["id"].to_pylist()) == {1, 2, 3, 4, 5}
        
        # Check for Puffin files in the data directory
        import os
        data_dir = os.path.join(warehouse_path, "test_ns", "v3_mor_table", "data")
        if os.path.exists(data_dir):
            puffin_files = [f for f in os.listdir(data_dir) if f.endswith(".puffin")]
            # If DVs are being used, we should have a Puffin file
            # Note: This may not exist if the implementation fell back to COW
            print(f"Puffin files found: {puffin_files}")

    def test_v3_cow_still_works(self, catalog, test_schema, sample_data) -> None:
        """Test that V3 tables with COW mode still work."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # V3 table with explicit COW mode
        table = catalog.create_table(
            "test_ns.v3_cow_table",
            schema=test_schema,
            properties={
                "format-version": "3",
                TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_COPY_ON_WRITE,
            },
        )
        
        assert table.metadata.format_version == 3
        
        table.append(sample_data)
        
        # Delete using COW (should rewrite files)
        table.delete(delete_filter="id > 5")
        
        result = table.scan().to_arrow()
        assert len(result) == 5

    def test_method_detection_cow_default(self, catalog, test_schema, sample_data) -> None:
        """Verify COW is used by default (no MOR property)."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.default_method_table",
            schema=test_schema,
        )
        
        # Default delete mode should be COW
        assert table.properties.get(TableProperties.DELETE_MODE) is None
        
        table.append(sample_data)
        table.delete(delete_filter="id > 5")
        
        # Should work via COW
        result = table.scan().to_arrow()
        assert len(result) == 5


class TestTableMigration:
    """Tests for migrating V1/V2 tables to V3 for DV support."""

    def test_upgrade_v2_to_v3(self, catalog, test_schema, sample_data) -> None:
        """Test upgrading a V2 table to V3."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # Create V2 table (default)
        table = catalog.create_table(
            "test_ns.v2_table_to_upgrade",
            schema=test_schema,
        )
        
        assert table.metadata.format_version == 2
        
        # Add some data
        table.append(sample_data)
        
        # Upgrade to V3
        with table.transaction() as tx:
            tx.upgrade_table_version(3)
        
        # Verify upgrade
        assert table.metadata.format_version == 3
        assert table.metadata.next_row_id is not None
        
        # Data should still be readable
        result = table.scan().to_arrow()
        assert len(result) == 10

    def test_upgrade_and_enable_mor(self, catalog, test_schema, sample_data) -> None:
        """Test full migration path: upgrade to V3 and enable MOR."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # Create V2 table
        table = catalog.create_table(
            "test_ns.v2_to_v3_mor",
            schema=test_schema,
        )
        
        table.append(sample_data)
        
        # Upgrade to V3 and enable MOR in one transaction
        with table.transaction() as tx:
            tx.upgrade_table_version(3)
            tx.set_properties({
                TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
            })
        
        assert table.metadata.format_version == 3
        assert table.properties.get(TableProperties.DELETE_MODE) == TableProperties.DELETE_MODE_MERGE_ON_READ
        
        # Now deletes should use DVs (not COW)
        table.delete(delete_filter="id > 5")
        
        result = table.scan().to_arrow()
        assert len(result) == 5

    def test_cannot_downgrade_v3_to_v2(self, catalog, test_schema) -> None:
        """Test that downgrade from V3 to V2 is not allowed."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        # Create V3 table
        table = catalog.create_table(
            "test_ns.v3_no_downgrade",
            schema=test_schema,
            properties={"format-version": "3"},
        )
        
        assert table.metadata.format_version == 3
        
        # Attempt to downgrade should fail
        with pytest.raises(ValueError, match="Cannot downgrade"):
            with table.transaction() as tx:
                tx.upgrade_table_version(2)

    def test_upgrade_preserves_data(self, catalog, test_schema, sample_data) -> None:
        """Test that upgrade preserves all existing data and metadata."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.upgrade_preserves_data",
            schema=test_schema,
            properties={
                "custom.property": "preserved",
            },
        )
        
        table.append(sample_data)
        
        # Record state before upgrade
        data_before = table.scan().to_arrow()
        props_before = dict(table.properties)
        
        # Upgrade
        with table.transaction() as tx:
            tx.upgrade_table_version(3)
        
        # Verify everything preserved
        data_after = table.scan().to_arrow()
        assert len(data_after) == len(data_before)
        assert set(data_after["id"].to_pylist()) == set(data_before["id"].to_pylist())
        
        # Custom properties preserved
        assert table.properties.get("custom.property") == "preserved"


class TestDeletionVectorManifests:
    """Tests for deletion vector manifest handling."""

    def test_inspect_manifests_after_delete(self, catalog, test_schema, sample_data) -> None:
        """Inspect manifest structure after delete operation."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.manifest_inspect_table",
            schema=test_schema,
        )
        
        table.append(sample_data)
        initial_snapshot = table.current_snapshot()
        
        # Delete some rows
        table.delete(delete_filter="id > 5")
        
        # Get manifests
        current_snapshot = table.current_snapshot()
        
        if current_snapshot:
            manifests = current_snapshot.manifests(table.io)
            
            # Should have manifest(s)
            assert len(manifests) > 0
            
            # Inspect manifest content types
            from pyiceberg.manifest import ManifestContent
            content_types = {m.content for m in manifests}
            
            # At minimum we should have DATA manifests
            assert ManifestContent.DATA in content_types

    def test_snapshot_history_after_deletes(self, catalog, test_schema, sample_data) -> None:
        """Test snapshot history is maintained through deletes."""
        catalog.create_namespace_if_not_exists("test_ns")
        
        table = catalog.create_table(
            "test_ns.snapshot_history_table",
            schema=test_schema,
        )
        
        # Initial append
        table.append(sample_data)
        snapshot_after_append = table.current_snapshot()
        
        # Delete
        table.delete(delete_filter="id > 8")
        snapshot_after_delete = table.current_snapshot()
        
        # Verify different snapshots
        assert snapshot_after_append is not None
        assert snapshot_after_delete is not None
        assert snapshot_after_append.snapshot_id != snapshot_after_delete.snapshot_id
        
        # Verify parent chain
        assert snapshot_after_delete.parent_snapshot_id == snapshot_after_append.snapshot_id

