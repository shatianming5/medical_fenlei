from __future__ import annotations

from dataclasses import dataclass

from medical_fenlei.constants import CLASS_ID_TO_NAME


@dataclass(frozen=True)
class TaskSpec:
    """
    Classification task spec.

    - For multiclass(6): `kind="multiclass"`, `num_classes=6`, and `pos_codes/neg_codes` are empty.
    - For binary: `kind="binary"`, `num_classes=2`, and codes refer to the original label codes (1..6).
    """

    name: str
    kind: str
    num_classes: int
    class_id_to_name: dict[int, str]
    neg_codes: tuple[int, ...] = ()
    pos_codes: tuple[int, ...] = ()

    def relevant_codes(self) -> set[int]:
        return set(self.neg_codes) | set(self.pos_codes)

    def neg_label_ids(self) -> tuple[int, ...]:
        return tuple(int(c) - 1 for c in self.neg_codes)

    def pos_label_ids(self) -> tuple[int, ...]:
        return tuple(int(c) - 1 for c in self.pos_codes)


SIX_CLASS = TaskSpec(
    name="six_class",
    kind="multiclass",
    num_classes=6,
    class_id_to_name={int(k): str(v) for k, v in CLASS_ID_TO_NAME.items()},
)


TASKS: dict[str, TaskSpec] = {
    # Multiclass
    SIX_CLASS.name: SIX_CLASS,
    # Binary tasks (user-defined)
    # 2) 正常 vs 患病（默认：患病=1..4；不包含“其他(6)”）
    "normal_vs_diseased": TaskSpec(
        name="normal_vs_diseased",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "正常", 1: "患病"},
        neg_codes=(5,),
        pos_codes=(1, 2, 3, 4),
    ),
    # 2b) 正常 vs 非正常（包含“其他(6)”）：推荐作为表征学习主任务
    "normal_vs_abnormal": TaskSpec(
        name="normal_vs_abnormal",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "正常", 1: "异常"},
        neg_codes=(5,),
        pos_codes=(1, 2, 3, 4, 6),
    ),
    # 3) 正常 vs 慢性化脓性中耳炎(1)
    "normal_vs_csoma": TaskSpec(
        name="normal_vs_csoma",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "正常", 1: CLASS_ID_TO_NAME[0]},
        neg_codes=(5,),
        pos_codes=(1,),
    ),
    # 4) 正常 vs 中耳胆脂瘤(2)
    "normal_vs_cholesteatoma": TaskSpec(
        name="normal_vs_cholesteatoma",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "正常", 1: CLASS_ID_TO_NAME[1]},
        neg_codes=(5,),
        pos_codes=(2,),
    ),
    # 5) 正常 vs 胆固醇肉芽肿(4)
    "normal_vs_cholesterol_granuloma": TaskSpec(
        name="normal_vs_cholesterol_granuloma",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "正常", 1: CLASS_ID_TO_NAME[3]},
        neg_codes=(5,),
        pos_codes=(4,),
    ),
    # 6) 正常 vs 分泌性中耳炎(3)
    "normal_vs_ome": TaskSpec(
        name="normal_vs_ome",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "正常", 1: CLASS_ID_TO_NAME[2]},
        neg_codes=(5,),
        pos_codes=(3,),
    ),
    # 7) 分泌性中耳炎(3) vs 胆固醇肉芽肿(4)
    "ome_vs_cholesterol_granuloma": TaskSpec(
        name="ome_vs_cholesterol_granuloma",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: CLASS_ID_TO_NAME[2], 1: CLASS_ID_TO_NAME[3]},
        neg_codes=(3,),
        pos_codes=(4,),
    ),
    # 8) 中耳胆脂瘤(2) vs 慢性化脓性中耳炎(1)
    "cholesteatoma_vs_csoma": TaskSpec(
        name="cholesteatoma_vs_csoma",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: CLASS_ID_TO_NAME[1], 1: CLASS_ID_TO_NAME[0]},
        neg_codes=(2,),
        pos_codes=(1,),
    ),
    # 胆脂瘤(2) vs 其他异常(1,3,4,6)：更贴近鉴别诊断（不含正常）
    "cholesteatoma_vs_other_abnormal": TaskSpec(
        name="cholesteatoma_vs_other_abnormal",
        kind="binary",
        num_classes=2,
        class_id_to_name={0: "其他异常", 1: CLASS_ID_TO_NAME[1]},
        neg_codes=(1, 3, 4, 6),
        pos_codes=(2,),
    ),
}


ALIASES: dict[str, str] = {
    "6": "six_class",
    "6class": "six_class",
    "multiclass": "six_class",
    "二分类_正常_vs_患病": "normal_vs_diseased",
    "正常vs患病": "normal_vs_diseased",
    "正常vs非正常": "normal_vs_abnormal",
    "正常vs异常": "normal_vs_abnormal",
    "正常vs慢性化脓性中耳炎": "normal_vs_csoma",
    "正常vs中耳胆脂瘤": "normal_vs_cholesteatoma",
    "正常vs胆固醇肉芽肿": "normal_vs_cholesterol_granuloma",
    "正常vs分泌性中耳炎": "normal_vs_ome",
    "分泌性中耳炎vs胆固醇肉芽肿": "ome_vs_cholesterol_granuloma",
    "中耳胆脂瘤vs慢性化脓性中耳炎": "cholesteatoma_vs_csoma",
    "胆脂瘤vs其他异常": "cholesteatoma_vs_other_abnormal",
}


def resolve_task(name: str) -> TaskSpec:
    key = str(name).strip()
    key = ALIASES.get(key, key)
    if key not in TASKS:
        known = ", ".join(sorted(TASKS.keys()))
        raise ValueError(f"unknown task: {name!r}. known: {known}")
    return TASKS[key]
