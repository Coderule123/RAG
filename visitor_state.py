import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


COMMON_SURNAME_CHARS = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉"
    "岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆"
    "萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮"
    "蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万"
    "支柯昝管卢莫经房裘缪解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢"
    "滑裴陆荣翁荀羊於惠甄麴家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷"
    "车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖武符刘景詹束龙叶幸司韶"
    "郜黎蓟薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴鬱胥能苍双闻莘党翟"
    "谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边燕冀郏浦尚农温别庄晏柴"
    "瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广"
    "禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养"
    "鞠须丰巢关蒯相查后荆红游竺权逑盖益桓公"
)

NAME_PATTERNS = [
    re.compile(r"(?:我叫|我是|名字叫|叫我|称呼我)\s*([一-龥]{2,4})"),
    re.compile(rf"^([{COMMON_SURNAME_CHARS}][一-龥]{{1,2}})$"),
]


def extract_name_from_query(query: str) -> Optional[str]:
    """
    从用户输入中提取姓名（占位实现）。

    TODO: 后续接入真实大模型，使用“提取用户姓名”提示词进行抽取。
    """
    text = (query or "").strip()
    if not text:
        return None

    for pattern in NAME_PATTERNS:
        matched = pattern.search(text)
        if matched:
            return matched.group(1).strip()
    return None


class VisitorStateStore:
    """访客状态存储：将视觉标识状态落盘到 JSON 文件。"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._save({})

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            normalized[key] = {
                "asked": bool(value.get("asked", False)),
                "pending_extract": bool(value.get("pending_extract", False)),
                "name": value.get("name"),
            }
        return normalized

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, vision_id: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        return data.get(vision_id)

    def mark_asked(self, vision_id: str) -> None:
        data = self._load()
        data[vision_id] = {"asked": True, "pending_extract": True, "name": None}
        self._save(data)

    def set_name(self, vision_id: str, name: str) -> None:
        data = self._load()
        state = data.get(vision_id, {})
        state["asked"] = True
        state["pending_extract"] = False
        state["name"] = name
        data[vision_id] = state
        self._save(data)

    def clear_pending(self, vision_id: str) -> None:
        data = self._load()
        state = data.get(vision_id, {})
        state["asked"] = True
        state["pending_extract"] = False
        state["name"] = state.get("name")
        data[vision_id] = state
        self._save(data)
