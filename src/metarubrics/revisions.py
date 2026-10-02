"""Content-addressed, training-only counterfactual rubric revisions.

Validation labels must come from a fixed independent reference panel, not
from rewards assigned by the rubric being selected. Evidence is supplied by
trusted callers; identities and expert approval are audit records, not auth.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from .profiles import profile, tau_cap

LABELS = {"INVARIANT", "TARGET_CHANGE", "WEIGHT_CHANGE", "DROPPED", "ADDED"}
ANCHORS = {"nice_to_have": 4., "should_have": 5., "must_have": 6.,
           "contraindication": 8., "not_applicable": 0.}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_snapshot(snapshot):
    require(snapshot.get("method") == "MetaRubrics" and snapshot.get("schema_version") == 1,
            "unsupported MetaRubrics snapshot")
    require(type(snapshot.get("version")) is int and snapshot["version"] >= 0, "invalid version")
    dataset = snapshot.get("dataset", "healthbench")
    anchors = profile(dataset)["anchors"]
    tau = snapshot.get("tau")
    require(isinstance(tau, dict), "tau must be a mapping")
    for cell, value in tau.items():
        require(cell in {s + "|" + l for s in ANCHORS for l in LABELS}, "invalid tau cell")
        require(type(value) in (int, float) and math.isfinite(value) and
                abs(value) <= tau_cap(dataset) + 1e-10, "tau exceeds bounds")
    for label in LABELS:
        magnitudes = [anchors[s] * math.exp(tau.get(s + "|" + label, 0.))
                      for s in ("nice_to_have", "should_have", "must_have", "contraindication")]
        require(all(a <= b + 1e-9 for a, b in zip(magnitudes, magnitudes[1:])), "severity order violated")
    require(isinstance(snapshot.get("revisions"), dict), "revisions must be a mapping")
    require(isinstance(snapshot.get("history"), list), "history must be a list")
    for sample_id, entry in snapshot["revisions"].items():
        require(isinstance(sample_id, str) and bool(sample_id), "invalid sample id")
        require(isinstance(entry, dict) and set(entry) == {"base_contract_sha256", "edits"},
                "invalid revision entry")
        base_hash = entry["base_contract_sha256"]
        require(isinstance(base_hash, str) and len(base_hash) == 64 and
                all(c in "0123456789abcdef" for c in base_hash), "invalid base contract hash")
        require(isinstance(entry["edits"], dict), "edits must be a mapping")
        for index, edit in entry["edits"].items():
            require(isinstance(index, str) and index.isascii() and index.isdigit() and
                    str(int(index)) == index, "invalid criterion index")
            require(isinstance(edit, dict) and set(edit) == {"criterion", "edit_label"},
                    "unexpected editable field")
            require(isinstance(edit["criterion"], str) and bool(edit["criterion"].strip()), "empty criterion")
            require(isinstance(edit["edit_label"], str) and edit["edit_label"] in LABELS, "invalid edit label")


def initial_snapshot(tau=None, dataset="healthbench"):
    result = dict(method="MetaRubrics", schema_version=1, version=0,
                  tau=dict(tau or {}), dataset=dataset, revisions={}, history=[])
    validate_snapshot(result)
    return result


def apply_revisions(contract, snapshot, split):
    """Return a copy; benchmark/validation contracts are never revised."""
    validate_snapshot(snapshot)
    result = deepcopy(contract)
    if split != "train":
        return result
    entry = snapshot["revisions"].get(contract["sample_id"])
    if entry is None:
        return result
    require(contract.get("side") == "twin", "only counterfactual twin contracts may be revised")
    require(digest(contract) == entry["base_contract_sha256"], "stale or mismatched base contract")
    for index, edit in entry["edits"].items():
        i = int(index)
        require(0 <= i < len(result["rubrics"]), "invalid criterion index")
        require(set(edit) == {"criterion", "edit_label"}, "unexpected editable field")
        require(isinstance(edit["criterion"], str) and edit["criterion"].strip(), "empty criterion")
        require(edit["edit_label"] in LABELS, "invalid edit label")
        result["rubrics"][i]["criterion"] = edit["criterion"]
        result["rubric_meta"][i]["edit_label"] = edit["edit_label"]
    return result


def proposal_request(contract, snapshot, proposal_case_ids, proposer_id):
    require(contract.get("side") == "twin", "expected counterfactual twin")
    return {
        "instruction": "Propose JSON edits to counterfactual criteria: target, applicability wording, "
        "or invariance label. Keep sign, severity, clinical anchors, exam items and original case fixed. "
        "Each edit has criterion_index, criterion, edit_label, rationale. All content edits require "
        "explicit model review and independent fixed-label validation before activation.",
        "sample_id": contract["sample_id"], "base_contract_sha256": digest(contract),
        "parent_snapshot_sha256": digest(snapshot), "proposer_id": proposer_id,
        "proposal_case_ids": proposal_case_ids,
        "current_contract": apply_revisions(contract, snapshot, "train"),
    }


def accept_revision(snapshot, contract, proposal, evidence, review, *, min_cases=2, min_gain=0.):
    """Select one proposal using paired agreement with fixed reference labels.

    Each validation row supplies case_id, reference_met, baseline_met and
    revised_met. Aggregate within case first. This is rubric fidelity validation,
    not a claim of downstream policy improvement. Negative criteria use the
    same met/not-met agreement; their reward sign stays frozen.
    """
    validate_snapshot(snapshot)
    require(type(min_cases) is int and min_cases >= 2, "min_cases must be >= 2")
    require(math.isfinite(min_gain) and min_gain >= 0, "invalid minimum gain")
    require(contract.get("side") == "twin", "only counterfactual twins may be revised")
    require(len(contract["rubrics"]) == len(contract["rubric_meta"]), "metadata mismatch")
    require(proposal["sample_id"] == contract["sample_id"], "wrong sample")
    require(proposal["base_contract_sha256"] == digest(contract), "stale base")
    require(proposal["parent_snapshot_sha256"] == digest(snapshot), "stale parent")
    proposal_hash = digest(proposal)
    require(review.get("proposal_sha256") == proposal_hash and review.get("approved") is True
            and review.get("review_kind") == "model_review"
            and review.get("model") == "gpt-5.4-mini"
            and bool(review.get("reviewer_id")) and bool(review.get("rationale")),
            "explicit gpt-5.4-mini model review required; not human clinical approval")
    require(evidence.get("proposal_sha256") == proposal_hash, "evidence not bound to proposal")
    require(evidence.get("split") == "train_validation" and
            evidence.get("source") == "fixed_reference_panel", "independent training validation required")
    identities = [proposal.get("proposer_id"), evidence.get("evaluator_id"), evidence.get("inner_judge_id")]
    require(all(isinstance(x, str) and x.strip() for x in identities) and len(set(identities)) == 3,
            "proposer, evaluator and inner judge must be distinct")
    require(bool(evidence.get("reference_panel_sha256")), "missing fixed reference panel provenance")
    require(bool(proposal.get("proposal_case_ids")), "missing proposal case provenance")
    rows = evidence.get("rows", [])
    cases, seen = {}, set()
    for row in rows:
        key = (row["case_id"], row["item_id"])
        require(key not in seen, "duplicate validation item")
        seen.add(key)
        require(row["case_id"] not in proposal["proposal_case_ids"] and
                row["case_id"] != contract["pair_id"], "proposal/validation case leakage")
        require(all(type(row[k]) is bool for k in ("reference_met", "baseline_met", "revised_met")),
                "validation grades must be booleans")
        cases.setdefault(row["case_id"], []).append(
            int(row["revised_met"] == row["reference_met"]) -
            int(row["baseline_met"] == row["reference_met"]))
    require(len(cases) >= min_cases, "insufficient independent validation cases")
    gain = sum(sum(v) / len(v) for v in cases.values()) / len(cases)
    require(gain > min_gain, "revision did not improve fixed-reference agreement")
    result = deepcopy(snapshot)
    entry = result["revisions"].setdefault(contract["sample_id"],
        {"base_contract_sha256": digest(contract), "edits": {}})
    require(entry["base_contract_sha256"] == digest(contract), "base changed across revisions")
    require(bool(proposal.get("edits")), "empty proposal")
    indices = set()
    for edit in proposal["edits"]:
        require(set(edit) == {"criterion_index", "criterion", "edit_label", "rationale"},
                "only criterion text and edit label can change")
        i = edit["criterion_index"]
        require(type(i) is int and 0 <= i < len(contract["rubrics"]) and i not in indices,
                "invalid or duplicate criterion index")
        indices.add(i)
        require(bool(edit["rationale"]), "missing rationale")
        entry["edits"][str(i)] = {k: edit[k] for k in ("criterion", "edit_label")}
    apply_revisions(contract, result, "train")
    result["version"] += 1
    result["history"].append(dict(kind="rubric_revision", parent_sha256=digest(snapshot),
        proposal=deepcopy(proposal), evidence=deepcopy(evidence), review=deepcopy(review),
        agreement_gain=gain, validation_cases=len(cases)))
    return result


def publish(snapshot, path):
    """Publish a complete immutable file atomically; never overwrite a segment."""
    validate_snapshot(snapshot)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".snapshot-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
