import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_healthbench import prepare


def row(pair_id, side, sample_id):
    contract = {
        "pair_id": pair_id,
        "side": side,
        "sample_id": sample_id,
    }
    return {
        "reward_model": {"ground_truth": json.dumps(contract)},
        "extra_info": {"split": "train"},
    }


class PrepareHealthBenchTest(unittest.TestCase):
    def test_adds_top_level_pair_id_to_original_schema(self):
        source_rows = [
            row("pair-a", "twin", "a-twin"),
            row("single-b", "orig", "b-orig"),
            row("pair-a", "orig", "a-orig"),
        ]
        original = pa.Table.from_pylist(source_rows)
        self.assertNotIn("pair_id", original.schema.names)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            output = root / "prepared"
            pq.write_table(original, source)

            audit = prepare(source, output)
            prepared = pq.read_table(output / "train.parquet")

        self.assertIn("pair_id", prepared.schema.names)
        self.assertEqual(prepared.column("pair_id").to_pylist(),
                         ["pair-a", "pair-a", "single-b"])
        self.assertEqual(audit["paired_cases"], 1)
        self.assertEqual(audit["unpaired_original_cases"], 1)

    def test_reuses_schema_that_already_has_pair_id(self):
        source_rows = [
            {**row("pair-a", "twin", "a-twin"), "pair_id": "pair-a"},
            {**row("pair-a", "orig", "a-orig"), "pair_id": "pair-a"},
        ]
        original = pa.Table.from_pylist(source_rows)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            output = root / "prepared"
            pq.write_table(original, source)
            prepare(source, output)
            prepared = pq.read_table(output / "train.parquet")

        self.assertEqual(prepared.schema.names.count("pair_id"), 1)
        self.assertEqual(prepared.column("pair_id").to_pylist(), ["pair-a", "pair-a"])


if __name__ == "__main__":
    unittest.main()
