from __future__ import annotations

from typing import Final

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.tasks import TaskSpec

DEFAULT_CLASS_PROMPTS_ZH: Final[dict[int, str]] = {
    # 0: 慢性化脓性中耳炎
    0: (
        "影像所见：鼓膜增厚/内陷或穿孔，鼓室与乳突可见软组织影或分泌物，乳突可呈硬化型或气化减少，"
        "听小骨形态可存留，骨皮质可毛糙但通常无明确锐利的侵蚀性骨质破坏。\n"
        "检查结论：慢性化脓性中耳炎。"
    ),
    # 1: 中耳胆脂瘤
    1: (
        "影像所见：中耳或上鼓室见软组织肿块影，边界清楚锐利，伴局灶性骨质吸收/破坏，"
        "可见听小骨链吸收或破坏，鼓室窦口/上鼓室扩大。\n"
        "检查结论：中耳胆脂瘤。"
    ),
    # 2: 分泌性中耳炎
    2: (
        "影像所见：鼓室及乳突气房内见液体密度影，可见气液平面或气泡征，"
        "听小骨链形态多正常，未见明确骨质破坏。\n"
        "检查结论：分泌性中耳炎。"
    ),
    # 3: 胆固醇肉芽肿（check.md 提供的标准描述 prompt）
    3: (
        "颞骨岩尖或中耳内的囊性病变，膨胀性生长，边缘光滑锐利，无骨质侵蚀性破坏，MRI T1高信号。\n"
        "检查结论：胆固醇肉芽肿。"
    ),
    # 4: 正常
    4: (
        "影像所见：双侧外耳道通畅，中耳鼓室及乳突气房气化良好，未见异常软组织影，"
        "听小骨链与周围骨质结构完整。\n"
        "检查结论：未见明显异常（正常）。"
    ),
    # 5: 其他
    5: (
        "影像所见：耳部异常改变不符合以上典型诊断特征，或为其他少见/混合型病变。\n"
        "检查结论：其他。"
    ),
}


def get_default_class_prompts_zh() -> dict[int, str]:
    """
    Return default Chinese prompts for each class id.

    Ensures all class ids in CLASS_ID_TO_NAME are present; falls back to
    a minimal prompt when missing.
    """
    out: dict[int, str] = {}
    for cid, name in sorted(CLASS_ID_TO_NAME.items()):
        if int(cid) in DEFAULT_CLASS_PROMPTS_ZH:
            out[int(cid)] = str(DEFAULT_CLASS_PROMPTS_ZH[int(cid)])
        else:
            out[int(cid)] = f"检查结论：{str(name)}。"
    return out


def get_task_prompts_zh(task: TaskSpec) -> dict[int, str]:
    """
    Build Chinese prompts for a classification task.

    - multiclass(6): returns per-class prompts (0..5) matching CLASS_ID_TO_NAME.
    - binary: returns prompts for output classes (0=neg, 1=pos) by concatenating
      the underlying 6-class prompts referenced by task.pos_codes/task.neg_codes.

    This enables check.md-style "text-guided prototypes" even for binary tasks.
    """
    task_kind = str(getattr(task, "kind", "")).strip().lower()
    num_classes = int(getattr(task, "num_classes", 0) or 0)

    base = get_default_class_prompts_zh()
    if task_kind != "binary":
        return base

    if num_classes != 2:
        # Fallback: minimal prompts from class names.
        out: dict[int, str] = {}
        names = getattr(task, "class_id_to_name", {}) or {}
        for i in range(int(num_classes)):
            out[int(i)] = f"检查结论：{str(names.get(int(i), f'类别{i}'))}。"
        return out

    def _join(ids: tuple[int, ...], *, fallback_name: str) -> str:
        parts = [str(base.get(int(i), "")).strip() for i in ids if str(base.get(int(i), "")).strip()]
        if parts:
            return "\n".join(parts).strip()
        return f"检查结论：{fallback_name}。"

    neg_ids = tuple(int(x) for x in getattr(task, "neg_label_ids", lambda: ())() or ())
    pos_ids = tuple(int(x) for x in getattr(task, "pos_label_ids", lambda: ())() or ())
    names = getattr(task, "class_id_to_name", {}) or {}
    neg_text = _join(neg_ids, fallback_name=str(names.get(0, "阴性")))
    pos_text = _join(pos_ids, fallback_name=str(names.get(1, "阳性")))
    return {0: neg_text, 1: pos_text}
