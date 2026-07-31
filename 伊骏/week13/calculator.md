# Skill: calculator
## Meta
id: calculator
name: 四则计算器
tags: ["math", "compute"]
description: 完成基础加减乘除运算
## Params
a: float
b: float
op: str
## Prompt
你是数学计算器，接收参数a、b与运算符op，计算表达式结果
## Code
```python
def run(a, b, op):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    raise ValueError("不支持的运算符")

# 下一步：如何接入 agent.py
## 1. Agent初始化Harness实例
```python
# agent.py
import os
from skill_harness import SkillHarness

# 路径：src上级目录是agent_memory_system，以此拼接skills路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKILLS_FOLDER = os.path.join(BASE_DIR, "skills")

# Agent内部持有harness实例
self.harness = SkillHarness(skills_root=SKILLS_FOLDER)