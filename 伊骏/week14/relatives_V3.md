# Skill: relatives 亲属称呼查询（V3完整版）
## Meta
skill_id: relatives
version: v3.0
tags: ["life", "relative", "query"]
desc: 递归解析亲属关系文本，输出标准亲属称谓，无法解析统一返回固定兜底文本
max_recursion_depth: 8
timeout_ms: 200
author: homework_student

## 入参Schema
```json
{
  "type": "object",
  "required": ["relation_text"],
  "properties": {
    "relation_text": {
      "type": "string",
      "minLength": 1,
      "maxLength": 128,
      "description": "亲属关系描述语句，可带/不带开头“我”，支持日常口语"
    }
  }
}

import re
from functools import lru_cache

class RelParser:
    # 口语/方言同义转换映射
    synonym_map = {
        "姥姥": "外婆",
        "姥爷": "外公",
        "老爹": "爸爸",
        "娘亲": "妈妈"
    }

    # 全量亲属关系树形库：祖辈/父母/平辈/姻亲/晚辈/表亲全覆盖
    base_map = {
        # 远祖辈
        "太爷爷": {"妻": "太奶奶"},
        "太奶奶": {},
        "外太外公": {"妻": "外太外婆"},
        "外太外婆": {},
        # 爷爷奶奶、外公外婆
        "爷爷": {"妻": "奶奶", "父": "太爷爷", "母": "太奶奶", "兄": "伯公", "弟": "叔公", "姐": "姑婆", "妹": "姑婆"},
        "奶奶": {},
        "外公": {"妻": "外婆", "父": "外太外公", "母": "外太外婆", "兄": "舅公", "弟": "舅公", "姐": "姨婆", "妹": "姨婆"},
        "外婆": {},
        # 父母主体
        "爸爸": {
            "父": "爷爷", "母": "奶奶",
            "兄": "伯伯", "弟": "叔叔", "姐": "姑姑", "妹": "姑姑",
            "妻": "妈妈"
        },
        "妈妈": {
            "父": "外公", "母": "外婆",
            "兄": "舅舅", "弟": "舅舅", "姐": "姨妈", "妹": "姨妈",
            "夫": "爸爸"
        },
        # 父辈亲属及配偶
        "伯伯": {"妻": "伯母", "子": "堂哥/堂弟", "女": "堂姐/堂妹"},
        "叔叔": {"妻": "婶婶", "子": "堂哥/堂弟", "女": "堂姐/堂妹"},
        "姑姑": {"夫": "姑父", "子": "表哥/表弟", "女": "表姐/表妹"},
        "舅舅": {"妻": "舅妈", "子": "表哥/表弟", "女": "表姐/表妹"},
        "姨妈": {"夫": "姨父", "子": "表哥/表弟", "女": "表姐/表妹"},
        # 同辈兄弟姐妹及配偶子女
        "哥哥": {"妻": "嫂子", "子": "侄子", "女": "侄女"},
        "弟弟": {"妻": "弟媳", "子": "侄子", "女": "侄女"},
        "姐姐": {"夫": "姐夫", "子": "外甥", "女": "外甥女"},
        "妹妹": {"夫": "妹夫", "子": "外甥", "女": "外甥女"},
        # 表亲延伸
        "表哥": {"父": "表舅/表伯/表叔", "母": "表姨"},
        "表弟": {"父": "表舅/表伯/表叔", "母": "表姨"},
        "表舅": {},
        # 自身晚辈
        "儿子": {"妻": "儿媳", "子": "孙子", "女": "孙女"},
        "女儿": {"夫": "女婿", "子": "外孙", "女": "外孙女"}
    }

    # 正则拆分 "A的B" 结构
    split_pat = re.compile(r"(.+?)的(.+)")
    # 无关人物黑名单，前置拦截
    black_words = {"同学", "老师", "同事", "校长", "火星", "月球", "奥特曼", "明星"}

    @classmethod
    def clean_text(cls, text: str) -> str:
        # 清除标点、空白字符，去除开头“我”
        txt = re.sub(r"[，。！？,\s]", "", text.strip())
        txt = txt.lstrip("我")
        # 统一方言口语为标准称谓
        for old_word, std_word in cls.synonym_map.items():
            txt = txt.replace(old_word, std_word)
        return txt

    @classmethod
    @lru_cache(maxsize=256)
    def parse_core(cls, clean_txt: str, depth: int = 0, max_depth: int = 8):
        # 递归深度超限直接返回空
        if depth >= max_depth:
            return None
        # 命中无关黑名单直接拦截
        for ban_word in cls.black_words:
            if ban_word in clean_txt:
                return None
        match_res = cls.split_pat.fullmatch(clean_txt)
        # 无“X的Y”结构，判定为基础亲属节点
        if not match_res:
            return clean_txt if clean_txt in cls.base_map else None
        main_part, sub_relation = match_res.groups()
        # 递归解析主体亲属
        root_rel = cls.parse_core(main_part, depth + 1, max_depth)
        if root_rel is None or root_rel not in cls.base_map:
            return None
        rel_child_map = cls.base_map[root_rel]
        if sub_relation not in rel_child_map:
            return None
        return rel_child_map[sub_relation]

    @classmethod
    def parse(cls, raw_text: str):
        # 全局异常捕获，任意报错统一返回None
        try:
            clean_str = cls.clean_text(raw_text)
            return cls.parse_core(clean_str)
        except Exception:
            return None

# 对外暴露技能主函数
def relatives(relation_text: str) -> str:
    parse_result = RelParser.parse(relation_text)
    return parse_result if parse_result is not None else "该关系无法查询"

# 本地测试入口
if __name__ == "__main__":
    test_cases = [
        "我妈妈的妈妈",
        "姥姥的丈夫",
        "爸爸的姐姐的丈夫",
        "妈妈的弟弟的表哥",
        "爸爸的同学的奶奶",
        "火星的姑姑",
        "妈妈的太奶奶"
    ]
    for case in test_cases:
        print(f"输入：{case} → {relatives(case)}")