#!/usr/bin/env python3
"""Prepare paired HealthBench training data and an initial MetaRubrics snapshot."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from metarubrics.revisions import ANCHORS, LABELS, initial_snapshot, publish


def prepare(source: Path, output: Path) -> dict:
    source_table = pq.read_table(source)
    rows = source_table.to_pylist()
    groups = defaultdict(list)
    for row in rows:
        contract = json.loads(row["reward_model"]["ground_truth"])
        groups[contract["pair_id"]].append(row)
    paired = [group for group in groups.values() if len(group) == 2]
    single = [group[0] for group in groups.values() if len(group) == 1]
    if len(paired) * 2 + len(single) != len(rows):
        raise ValueError("a pair_id occurs more than twice")
    ordered = [row for pair in paired for row in sorted(pair, key=lambda row: json.loads(row["reward_model"]["ground_truth"])["side"])] + single
    for row in ordered:
        row["pair_id"] = json.loads(row["reward_model"]["ground_truth"])["pair_id"]
    if any(row.get("extra_info", {}).get("split") != "train" for row in ordered):
        raise ValueError("training parquet contains a non-training split")
    output.mkdir(parents=True, exist_ok=True)
    output_schema = (source_table.schema if "pair_id" in source_table.schema.names
                     else source_table.schema.append(pa.field("pair_id", pa.string())))
    pq.write_table(pa.Table.from_pylist(ordered, schema=output_schema), output / "train.parquet")
    tau = {severity + "|" + label: 0.0 for severity in ANCHORS for label in LABELS}
    publish(initial_snapshot(tau, "healthbench"), output / "snapshot-0.json")
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(ordered),
        "paired_cases": len(paired),
        "unpaired_original_cases": len(single),
        "shuffle": False,
    }
    (output / "dataset_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
