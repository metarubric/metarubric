"""Fixed dataset semantics, explicitly carried by the immutable snapshot."""
import math

PROFILES = {
    "healthbench": {
        "anchors": dict(not_applicable=0., nice_to_have=4., should_have=5., must_have=6., contraindication=8.),
        "data_sources": {"acre_healthbench_twin", "acre_healthbench_indomain"},
        "reward_mode": "legacy_scalar",
    },
}


def profile(dataset="healthbench"):
    if dataset not in PROFILES:
        raise ValueError("unsupported MetaRubrics dataset")
    return PROFILES[dataset]


def tau_cap(dataset="healthbench"):
    anchors = profile(dataset)["anchors"]
    values = [anchors[k] for k in ("nice_to_have", "should_have", "must_have", "contraindication")]
    return math.log(min(b / a for a, b in zip(values, values[1:])))
