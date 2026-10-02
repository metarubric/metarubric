"""Offline outer-loop boundary commands; no implicit policy or API launch."""
import argparse
import json
from pathlib import Path
from .revisions import (accept_revision, apply_revisions, digest, initial_snapshot,
                        proposal_request, publish, validate_snapshot)
from .weights import group_errors, update_tau


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(prog="metarubrics")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("output")
    init.add_argument("--dataset", choices=["healthbench"], default="healthbench")
    for name in ("request", "accept", "apply", "weights"):
        p = sub.add_parser(name)
        p.add_argument("snapshot")
        if name != "weights":
            p.add_argument("contract", help="original JSON training contract")
        p.add_argument("output")
        if name == "request":
            p.add_argument("--proposer", required=True)
            p.add_argument("--proposal-case", action="append", required=True)
        if name == "accept":
            for field in ("proposal", "evidence", "review"):
                p.add_argument("--" + field, required=True)
            p.add_argument("--min-cases", type=int, default=2)
            p.add_argument("--min-gain", type=float, default=0.)
        if name == "apply":
            p.add_argument("--split", choices=["train", "validation", "test"], required=True)
        if name == "weights":
            p.add_argument("--observations", required=True)
            p.add_argument("--eta", type=float, default=.1)
    args = parser.parse_args()
    if args.command == "init":
        from .revisions import ANCHORS, LABELS
        publish(initial_snapshot({s + "|" + l: 0. for s in ANCHORS for l in LABELS}, dataset=args.dataset), args.output)
        return
    snapshot = read(args.snapshot)
    validate_snapshot(snapshot)
    if args.command == "weights":
        observations = read(args.observations)
        stats = group_errors(observations)
        updated = dict(snapshot, version=snapshot["version"] + 1,
                       tau=update_tau(snapshot["tau"], stats, args.eta, dataset=snapshot.get("dataset", "healthbench")))
        updated["history"] = snapshot["history"] + [dict(kind="weight_update",
            parent_sha256=digest(snapshot), observations_sha256=digest(observations),
            stats=stats, eta=args.eta)]
        publish(updated, args.output)
        return
    contract = read(args.contract)
    if args.command == "accept":
        publish(accept_revision(snapshot, contract, read(args.proposal), read(args.evidence),
            read(args.review), min_cases=args.min_cases, min_gain=args.min_gain), args.output)
        return
    if args.command == "request":
        result = proposal_request(contract, snapshot, args.proposal_case, args.proposer)
    else:
        result = apply_revisions(contract, snapshot, args.split)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
