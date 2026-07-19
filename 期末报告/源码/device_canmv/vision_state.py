"""
Stage 1 视觉状态机 — CanMV K230D
解析 YOLOv8 检测结果，进行连续时间确认，输出稳定学习状态。

状态定义：
    FOCUS   — person 存在 + phone 不存在 + book 存在，持续确认
    PHONE   — person 存在 + phone 存在，持续确认
    PERSON  — person 存在 + 无书无手机 (Stage 2 新增)，持续确认
    AWAY    — person 不存在，持续确认
    UNKNOWN — 检测结果不足以判断，或处于过渡期

重要：
    如果 res 的结构和当前固件不一致，请先在 main.py 中设 DEBUG_PRINT_RES=1，
    查看串口打印的实际格式，再修改 parse_detections() 的解析逻辑。
"""

import time
from config import (
    LABELS,
    PHONE_CONFIRM_SECONDS,
    AWAY_CONFIRM_SECONDS,
    FOCUS_CONFIRM_SECONDS,
    PERSON_CONFIRM_SECONDS,
    DEBUG_PRINT_RES,
)
from utils import debug_print


# ============================================================
# 状态常量
# ============================================================
STATE_FOCUS = "FOCUS"
STATE_PHONE = "PHONE"
STATE_PERSON = "PERSON"
STATE_AWAY = "AWAY"
STATE_UNKNOWN = "UNKNOWN"


# ============================================================
# 检测结果解析
# ============================================================
def parse_detections(res):
    """
    解析 YOLOv8.run() 返回的检测结果。

    实测 K230D CanMV 0.4.0 固件的 YOLOv8 输出格式:
        res = [boxes_list, class_ids_list, scores_list]
        例: [[array([121,180,101,91])], [1], [0.2888184]]
        - boxes_list:    每个 box 是 [x1, y1, x2, y2] (归一化前)
        - class_ids_list: 整数, 对应 LABELS 索引
        - scores_list:    0~1 之间的浮点数

    兼容其他格式 (格式 A/B/C)。

    返回:
        dict: {
            "person": int,   # 检测到的 person 数量
            "phone": int,
            "book": int,
            "laptop": int,
            "raw": list,     # 原始检测框列表
        }
    """
    result = {"person": 0, "phone": 0, "book": 0, "laptop": 0, "raw": []}

    if res is None:
        return result

    # 格式 D (实测): [boxes, class_ids, scores] — 3 元素列表
    if isinstance(res, (list, tuple)) and len(res) == 3:
        try:
            boxes = res[0]
            elem1 = res[1]
            elem2 = res[2]
            # 通过元素类型区分: class_id 是 int, score 是 0~1 float
            # 尝试: elem1 是 class_ids, elem2 是 scores
            if (isinstance(boxes, (list, tuple)) and
                isinstance(elem1, (list, tuple)) and
                isinstance(elem2, (list, tuple))):
                # 试两种顺序
                # 先试 [boxes, class_ids, scores] (实测顺序)
                if (len(elem1) > 0 and isinstance(elem1[0], (int, float)) and
                    len(elem2) > 0 and isinstance(elem2[0], (int, float))):
                    e1_is_class = all(isinstance(x, int) or (isinstance(x, float) and x == int(x)) for x in elem1)
                    e2_is_score = all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in elem2)
                    if e1_is_class and e2_is_score:
                        # 格式: [boxes, class_ids, scores]
                        for i, box in enumerate(boxes):
                            cls_id = int(elem1[i]) if i < len(elem1) else -1
                            score = float(elem2[i]) if i < len(elem2) else 0.0
                            cls_name = _cls_id_to_name(cls_id)
                            if cls_name:
                                result[cls_name] += 1
                            result["raw"].append({
                                "box": box, "score": score, "class_id": cls_id
                            })
                        return result
                    # 试反向: [boxes, scores, class_ids]
                    e1_is_score = all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in elem1)
                    e2_is_class = all(isinstance(x, int) or (isinstance(x, float) and x == int(x)) for x in elem2)
                    if e1_is_score and e2_is_class:
                        for i, box in enumerate(boxes):
                            score = float(elem1[i]) if i < len(elem1) else 0.0
                            cls_id = int(elem2[i]) if i < len(elem2) else -1
                            cls_name = _cls_id_to_name(cls_id)
                            if cls_name:
                                result[cls_name] += 1
                            result["raw"].append({
                                "box": box, "score": score, "class_id": cls_id
                            })
                        return result
        except Exception:
            pass

    # 格式 A/B: 列表的列表 (每个子列表 [x1,y1,x2,y2,score,class_id] 等)
    if isinstance(res, (list, tuple)):
        dets = list(res)
        if len(dets) == 0:
            return result

        first = dets[0]

        if isinstance(first, (list, tuple)):
            if len(first) >= 6:
                a = first[0]
                b = first[1]
                is_format_b = (isinstance(a, (int, float)) and a < len(LABELS) and
                               isinstance(b, float) and 0.0 <= b <= 1.0)
            else:
                is_format_b = False

            for det in dets:
                try:
                    if is_format_b:
                        cls_id = int(det[0])
                        score = float(det[1])
                        box = det[2:6]
                    else:
                        box = det[0:4]
                        score = float(det[4])
                        cls_id = int(det[5])

                    cls_name = _cls_id_to_name(cls_id)
                    if cls_name:
                        result[cls_name] += 1
                    result["raw"].append({
                        "box": box, "score": score, "class_id": cls_id
                    })
                except (IndexError, ValueError, TypeError):
                    continue

        return result

    if DEBUG_PRINT_RES:
        print("[VISION_STATE] Unknown res format, type:", type(res), "value:", res)

    return result


def _cls_id_to_name(cls_id):
    """类别 ID → 名称，超出范围返回 None。"""
    if 0 <= cls_id < len(LABELS):
        return LABELS[cls_id]
    return None


# ============================================================
# 学习状态机
# ============================================================
class StudyStateMachine:
    """
    基于连续帧检测结果的学习状态机。

    核心逻辑：
        - 每个候选状态需要持续确认超过阈值秒数才会正式切换
        - 如果中途检测条件变化，重新计时
        - 避免状态在 FOCUS ↔ PHONE ↔ AWAY 之间疯狂跳变
    """

    def __init__(self):
        self._current_state = STATE_UNKNOWN

        # 候选状态计时：记录状态条件首次满足的时间戳 (ticks_ms)
        self._candidate_state = STATE_UNKNOWN
        self._candidate_start_ms = 0

        # 去抖阈值 (ms)
        self._focus_ms = FOCUS_CONFIRM_SECONDS * 1000
        self._phone_ms = PHONE_CONFIRM_SECONDS * 1000
        self._away_ms = AWAY_CONFIRM_SECONDS * 1000
        self._person_ms = PERSON_CONFIRM_SECONDS * 1000

    # ---- 属性 ----
    @property
    def state(self):
        return self._current_state

    @property
    def candidate(self):
        return self._candidate_state

    # ---- 核心更新 ----
    def update(self, person_count, phone_count, book_count, laptop_count):
        """
        每帧调用，根据当前计数更新状态机。

        参数:
            person_count, phone_count, book_count, laptop_count: 各类检测数量
        返回:
            str: 当前确认的状态 (FOCUS/PHONE/AWAY/UNKNOWN)
        """
        now = time.ticks_ms()

        # 1) 判断当前帧的瞬时状态
        instant = self._classify_instant(person_count, phone_count,
                                         book_count, laptop_count)

        # 2) 如果瞬时状态和候选状态不同，重置候选计时
        if instant != self._candidate_state:
            self._candidate_state = instant
            self._candidate_start_ms = now
            debug_print(2, "candidate ->", instant,
                        "person:", person_count, "phone:", phone_count,
                        "book:", book_count, "laptop:", laptop_count)
            return self._current_state

        # 3) 检查候选状态是否已经持续超过阈值
        elapsed = time.ticks_diff(now, self._candidate_start_ms)
        required = self._required_ms(instant)

        if elapsed >= required and instant != self._current_state:
            old = self._current_state
            self._current_state = instant
            # 总是打印状态变化 (不受 DEBUG_MODE 影响)
            print("[STATE]", old, "->", instant, "(elapsed:", elapsed, "ms)")

        return self._current_state

    def _classify_instant(self, person, phone, book, laptop):
        """根据各类计数判定瞬时状态。

        优先级: PHONE > AWAY(无人) > FOCUS(有书) > PERSON(只有人)
        """
        has_person = person > 0
        has_phone = phone > 0
        has_study_item = (book + laptop) > 0

        if has_phone:
            # 任何时候检测到手机都是 PHONE (不管人在不在)
            return STATE_PHONE
        elif not has_person:
            # 没人 + 没手机 = AWAY
            return STATE_AWAY
        elif has_study_item:
            return STATE_FOCUS
        elif has_person:
            return STATE_PERSON
        else:
            return STATE_UNKNOWN

    def _required_ms(self, state):
        if state == STATE_FOCUS:
            return self._focus_ms
        elif state == STATE_PHONE:
            return self._phone_ms
        elif state == STATE_AWAY:
            return self._away_ms
        elif state == STATE_PERSON:
            return self._person_ms
        return 0

    def reset(self):
        self._current_state = STATE_UNKNOWN
        self._candidate_state = STATE_UNKNOWN
        self._candidate_start_ms = 0
