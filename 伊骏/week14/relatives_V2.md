# Skill: relatives
## 功能
解析亲属关系文本，以用户视角返回标准称谓；解析失败输出：该关系无法查询
## 入参
relation_text:str,必填，亲属关系描述文本
## 输出规则
1. 自动省略开头“我”也可识别；
2. 递归拆解多层嵌套关系链；
3. 无匹配/解析异常统一返回固定文本，不编造称谓。
## 示例
输入：妈妈的妈妈 → 外婆
输入：爸爸的弟弟的儿子 → 堂哥/堂弟
输入：月球的姑妈 → 该关系无法查询
## 优化后代码
```python
import re

class RelParser:
    # 基础亲属单元映射，精简键值减少内存占用
    base_map = {
        "爷爷": {"父": "爸爸", "母": "奶奶"},
        "奶奶": {"父": "伯伯/叔叔", "母": "姑姑"},
        "外公": {"父": "舅舅", "母": "姨妈"},
        "外婆": {},
        "爸爸": {"父": "爷爷", "母": "奶奶", "兄": "伯伯", "弟": "叔叔", "姐": "姑姑", "妹": "姑姑"},
        "妈妈": {"父": "外公", "母": "外婆", "兄": "舅舅", "弟": "舅舅", "姐": "姨妈", "妹": "姨妈"},
        "伯伯": {"子": "堂哥/堂弟", "女": "堂姐/堂妹"},
        "叔叔": {"子": "堂哥/堂弟", "女": "堂姐/堂妹"},
        "姑姑": {"子": "表哥/表弟", "女": "表姐/表妹"},
        "舅舅": {"子": "表哥/表弟", "女": "表姐/表妹"},
        "姨妈": {"子": "表哥/表弟", "女": "表姐/表妹"},
    }
    # 分词规则：拆分A的B结构
    split_pat = re.compile(r"(.+?)的(.+)")

    @classmethod
    def parse(cls, text: str):
        # 预处理：去除开头多余"我"、空白
        clean_txt = text.strip().lstrip("我").strip()
        match = cls.split_pat.fullmatch(clean_txt)
        # 无"X的Y"结构，判定为基础亲属节点
        if not match:
            if clean_txt in cls.base_map:
                return clean_txt
            return None
        # 递归拆解：左主体 + 后缀关系
        main_part, sub_rel = match.groups()
        root_name = cls.parse(main_part)
        if root_name is None or root_name not in cls.base_map:
            return None
        rel_dict = cls.base_map[root_name]
        if sub_rel not in rel_dict:
            return None
        return rel_dict[sub_rel]

def relatives(relation_text: str) -> str:
    res = RelParser.parse(relation_text)
    return res if res is not None else "该关系无法查询"

if __name__ == "__main__":
    test_list = [
        "我妈妈的妈妈",
        "爸爸的弟弟的女儿",
        "妈妈的姐姐的儿子",
        "火星的舅舅",
        "爸爸的哥哥的妈妈"
    ]
    for t in test_list:
        print(f"{t} → {relatives(t)}")