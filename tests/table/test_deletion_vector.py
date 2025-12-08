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
"""Tests for deletion vector support."""

import tempfile
import uuid
from io import BytesIO

import pyarrow as pa
import pytest
from pyroaring import BitMap

from pyiceberg.io import load_file_io
from pyiceberg.manifest import DataFileContent, FileFormat
from pyiceberg.table.deletion_vector import (
    DeletionVectorResult,
    write_deletion_vectors,
)
from pyiceberg.table.puffin import PuffinFile


class TestWriteDeletionVectors:
    """Test writing deletion vectors to Puffin files."""

    def test_write_single_deletion_vector(self, tmp_path) -> None:
        """Write a single deletion vector and verify structure."""
        # Create a mock file IO
        io = load_file_io({"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"})
        
        # Create deletion positions
        positions = {
            f"file://{tmp_path}/data/file1.parquet": BitMap([1, 5, 10, 15, 20]),
        }
        
        # Mock table metadata (we just need location)
        class MockMetadata:
            location = f"file://{tmp_path}"
        
        result = write_deletion_vectors(
            io=io,
            table_metadata=MockMetadata(),
            deletion_positions=positions,
        )
        
        assert isinstance(result, DeletionVectorResult)
        assert len(result.delete_files) == 1
        assert result.total_deleted_rows == 5
        assert len(result.referenced_data_files) == 1
        
        # Verify the delete file structure
        delete_file = result.delete_files[0]
        assert delete_file.content == DataFileContent.POSITION_DELETES
        assert delete_file.file_format == FileFormat.PUFFIN
        assert delete_file.record_count == 5

    def test_write_multiple_deletion_vectors(self, tmp_path) -> None:
        """Write deletion vectors for multiple files."""
        io = load_file_io({"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"})
        
        positions = {
            f"file://{tmp_path}/data/file1.parquet": BitMap([1, 2, 3]),
            f"file://{tmp_path}/data/file2.parquet": BitMap([10, 20, 30, 40]),
            f"file://{tmp_path}/data/file3.parquet": BitMap([100, 200]),
        }
        
        class MockMetadata:
            location = f"file://{tmp_path}"
        
        result = write_deletion_vectors(
            io=io,
            table_metadata=MockMetadata(),
            deletion_positions=positions,
        )
        
        assert len(result.delete_files) == 3
        assert result.total_deleted_rows == 9  # 3 + 4 + 2
        assert len(result.referenced_data_files) == 3

    def test_write_empty_deletion_vectors(self, tmp_path) -> None:
        """Empty positions map returns empty result."""
        io = load_file_io({"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"})
        
        class MockMetadata:
            location = f"file://{tmp_path}"
        
        result = write_deletion_vectors(
            io=io,
            table_metadata=MockMetadata(),
            deletion_positions={},
        )
        
        assert len(result.delete_files) == 0
        assert result.total_deleted_rows == 0

    def test_skip_empty_positions(self, tmp_path) -> None:
        """Files with no deleted positions are skipped."""
        io = load_file_io({"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"})
        
        positions = {
            f"file://{tmp_path}/data/file1.parquet": BitMap([1, 2, 3]),
            f"file://{tmp_path}/data/file2.parquet": BitMap(),  # Empty
        }
        
        class MockMetadata:
            location = f"file://{tmp_path}"
        
        result = write_deletion_vectors(
            io=io,
            table_metadata=MockMetadata(),
            deletion_positions=positions,
        )
        
        # Only one file should have a delete file (file2 was empty)
        assert len(result.delete_files) == 1
        assert result.total_deleted_rows == 3

    def test_puffin_file_readable(self, tmp_path) -> None:
        """Verify the written Puffin file can be read back."""
        io = load_file_io({"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"})
        
        data_path = f"file://{tmp_path}/data/myfile.parquet"
        positions = {
            data_path: BitMap([5, 10, 15, 20, 25]),
        }
        
        class MockMetadata:
            location = f"file://{tmp_path}"
        
        result = write_deletion_vectors(
            io=io,
            table_metadata=MockMetadata(),
            deletion_positions=positions,
        )
        
        # Read back the Puffin file
        puffin_path = result.delete_files[0].file_path
        with io.new_input(puffin_path).open() as f:
            puffin = PuffinFile(f.read())
        
        # Verify the content
        vectors = puffin.to_vector()
        assert data_path in vectors
        
        # Extract positions from ChunkedArray
        read_positions = set()
        for chunk in vectors[data_path].chunks:
            for val in chunk:
                read_positions.add(val.as_py() & 0xFFFFFFFF)
        
        assert read_positions == {5, 10, 15, 20, 25}

