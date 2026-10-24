"""The not_a_sign crops of the classifier leave out the likely signs of the frames whose reviewer did not box the
other traffic signs (an AI before training.OTHER_SIGN_RULE_AT): the other-sign proposals it removed, and the
cutter's surest boxes there."""
from __future__ import annotations

from collections import Counter

from nnslr_tools.chain import MINE_SIGN_SCORE, frame_candidates
from nnslr_tools.training import BOXES_OTHER_SIGNS, _classifier_negatives


def _removed(family: str, x: float) -> dict:
    return {"annotation_id": f"{family}-{x}", "image_path": "frames/a.png", "image_sha256": "a" * 64,
            "bbox_xyxy": [x, 0, x + 10, 10], "sign_family": family, "review_state": "rejected"}


def _frame(tier: str, rejected: list[dict]) -> dict:
    return {"split": "train", "tier": tier, "label_basis": "ai_review", "kept": [{"bbox_xyxy": [500, 0, 510, 10]}],
            "rejected": rejected}


def test_removed_other_sign_proposals_count_only_where_other_signs_were_boxed():
    rejected = [_removed("other_sign", 0), _removed("maximum_speed", 100)]
    skipped: Counter = Counter()
    found = _classifier_negatives([_frame(tier, rejected) for tier in ("human", "ai_after_rule", "ai_before_rule",
                                                                       "other")], skipped)
    kept = Counter((item["tier"], item["proposal_family"]) for item in found)
    assert kept == {("human", "other_sign"): 1, ("human", "maximum_speed"): 1,
                    ("ai_after_rule", "other_sign"): 1, ("ai_after_rule", "maximum_speed"): 1,
                    ("ai_before_rule", "maximum_speed"): 1, ("other", "maximum_speed"): 1}
    assert skipped["rejected_other_sign_before_rule"] == 2
    assert set(BOXES_OTHER_SIGNS) == {"human", "ai_after_rule"}


def test_mining_drops_the_surest_boxes_only_where_other_signs_were_not_boxed():
    predictions = [{"box": [100 * n, 0, 100 * n + 10, 10], "score": score} for n, score in enumerate((.6, .3, .29, .1))]
    for tier in BOXES_OTHER_SIGNS:
        free, best = frame_candidates({"kept": [], "rejected": [], "tier": tier}, predictions, per_frame=3,
                                      sign_score=MINE_SIGN_SCORE)
        assert free == 4 and [item["score"] for item in best] == [.6, .3, .29]
    for tier in ("ai_before_rule", "other"):
        frame = {"kept": [], "rejected": [], "tier": tier}
        free, best = frame_candidates(frame, predictions, per_frame=3, sign_score=MINE_SIGN_SCORE)
        assert free == 2 and [item["score"] for item in best] == [.29, .1]
        # Without sign_score every free box counts, as before.
        assert frame_candidates(frame, predictions, per_frame=3)[0] == 4
