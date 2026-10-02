#!/usr/bin/env python3
"""Mask complete GRPO groups containing an invalid external-judge result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FailureIsolationReport:
    invalid_rollouts: int
    invalid_groups: int
    valid_groups: int


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"judge_valid must be Boolean, got {type(value).__name__}")


def isolate_invalid_judge_groups(
    *,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor,
    group_ids: Sequence[Any],
    judge_valid: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor, FailureIsolationReport]:
    """Neutralize every rollout in a GRPO group containing a judge failure.

    A judge failure is missing supervision, not a zero-quality answer.  The
    complete prompt group is therefore removed from both reward and policy
    loss while all other groups in the optimizer batch remain unchanged.
    """

    if response_mask.ndim != 2 or token_level_scores.shape != response_mask.shape:
        raise ValueError("response_mask and token_level_scores must have the same 2D shape")
    if len(group_ids) != response_mask.shape[0] or len(judge_valid) != response_mask.shape[0]:
        raise ValueError("group_ids and judge_valid must align with the rollout batch")

    normalized_valid = np.asarray([_as_bool(value) for value in judge_valid], dtype=bool)
    normalized_groups = np.asarray([str(value) for value in group_ids], dtype=object)
    invalid_groups = set(normalized_groups[~normalized_valid].tolist())
    keep = np.asarray([group not in invalid_groups for group in normalized_groups], dtype=bool)
    valid_groups = set(normalized_groups[keep].tolist())
    if not valid_groups:
        raise RuntimeError("all GRPO groups have invalid judge supervision")

    keep_tensor = torch.as_tensor(keep, device=response_mask.device, dtype=torch.bool)
    isolated_mask = response_mask.clone()
    isolated_scores = token_level_scores.clone()
    isolated_mask[~keep_tensor] = 0
    isolated_scores[~keep_tensor] = 0
    return (
        isolated_mask,
        isolated_scores,
        FailureIsolationReport(
            invalid_rollouts=int((~normalized_valid).sum()),
            invalid_groups=len(invalid_groups),
            valid_groups=len(valid_groups),
        ),
    )


def isolate_gated_groups(
    *,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor,
    group_ids: Sequence[Any],
    joint_gated: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor, FailureIsolationReport]:
    """Neutralize every rollout in a GRPO group the confidence gate withheld.

    A gated group is not a judge failure: the judge succeeded but its ordering
    was too close to trust (below_min_spread or a tie), so joint_group_reward
    withholds the rubric channel. Historically the group still trained, on an
    S/H residue roughly 56x smaller than an ungated group's delivered-score
    spread, which norm_adv_by_std_in_grpo then renormalizes to unit variance --
    the same gradient weight as a group carrying a confident rubric ordering,
    on noise. This zeroes gated groups out of both reward and policy loss the
    same way isolate_invalid_judge_groups zeroes judge failures, so remaining
    gradient comes only from groups the judge actually distinguished.
    """

    if response_mask.ndim != 2 or token_level_scores.shape != response_mask.shape:
        raise ValueError("response_mask and token_level_scores must have the same 2D shape")
    if len(group_ids) != response_mask.shape[0] or len(joint_gated) != response_mask.shape[0]:
        raise ValueError("group_ids and joint_gated must align with the rollout batch")

    normalized_gated = np.asarray([_as_bool(value) for value in joint_gated], dtype=bool)
    normalized_groups = np.asarray([str(value) for value in group_ids], dtype=object)
    gated_groups = set(normalized_groups[normalized_gated].tolist())
    keep = np.asarray([group not in gated_groups for group in normalized_groups], dtype=bool)
    valid_groups = set(normalized_groups[keep].tolist())
    if not valid_groups:
        raise RuntimeError("all GRPO groups in this batch were gated")

    keep_tensor = torch.as_tensor(keep, device=response_mask.device, dtype=torch.bool)
    isolated_mask = response_mask.clone()
    isolated_scores = token_level_scores.clone()
    isolated_mask[~keep_tensor] = 0
    isolated_scores[~keep_tensor] = 0
    return (
        isolated_mask,
        isolated_scores,
        FailureIsolationReport(
            invalid_rollouts=int(normalized_gated.sum()),
            invalid_groups=len(gated_groups),
            valid_groups=len(valid_groups),
        ),
    )
