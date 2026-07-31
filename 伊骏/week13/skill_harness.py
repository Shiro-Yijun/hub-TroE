import os
import re
import logging
from dataclasses import dataclass
from typing import Dict,List,Optional,Any
import yaml

# -------------------------- 日志配置 --------------------------
logger = logging.getLogger("skill_harness")

# -------------------------- 数据模型定义 --------------------------
@dataclass
class SkillMeta:
    """【轻量元数据】启动阶段只加载这个！渐进式核心：仅花名册，不加载完整提示词"""
    skill_id: str
    name: str
    tags: List[str]
    description: str
    file_path: str


@dataclass
class Skill:
    """完整Skill实例：只有按需加载时才构造"""
    meta: SkillMeta
    prompt_template: str
    parms_schema: Dict[str,str]
    code_block: Optional[str] = None    # 可选内嵌执行代码


# -------------------------- 核心Harness框架 --------------------------
class SkillHarness:
    def __init__(self, skills_root: str):
        self.skills_root = skills_root
        # 注册表：启动扫描填充，只存轻量Meta（渐进加载关键）
        self._registry: Dict[str, SkillMeta] = {}
        # 缓存池：已经完整加载实例的Skill，避免重复磁盘IO
        self.loaded_skill_cache: Dict[str, Skill] = {}

        # 初始化：扫描所有skill，注册元信息（不加载完整内容！）
        self.scan_and_register_skills()

    def scan_and_register_skills(self) -> None:
        """
        启动执行：扫描skills目录，只解析元信息，完成【注册】
        ❗不会读取prompt、代码等大块内容，实现启动轻量化
        """
        logger.info("开始扫描Skill目录，注册元信息（渐进加载模式）")
        self._registry.clear()

        for filename in os.listdir(self.skills_root):
            if not filename.endswith(".md"):
                continue
            file_path = os.path.join(self.skills_root, filename)
            try:
                meta = self._parse_skill_meta_only(file_path)
                self._registry[meta.skill_id] = meta
                logger.debug(f"成功注册Skill元信息: {meta.skill_id}")
            except Exception as e:
                logger.error(f"解析Skill元数据失败 {file_path}, err:{str(e)}")

        logger.info(f"Skill注册完成，共注册 {len(self._registry)} 个技能（尚未加载完整实例）")

    def _parse_skill_meta_only(self, file_path: str) -> SkillMeta:
        """只读取文件头部Meta块，不加载后续Prompt/Code，轻量化解析"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 匹配顶部 ## Meta yaml块
        meta_match = re.search(r"## Meta\n(.*?)(?=\n## |$)", content, re.DOTALL)
        if not meta_match:
            raise ValueError(f"Skill文件{file_path}缺少## Meta区块")
        meta_raw = yaml.safe_load(meta_match.group(1))
        return SkillMeta(
            skill_id=meta_raw["id"],
            name=meta_raw["name"],
            tags=meta_raw["tags"],
            description=meta_raw["description"],
            file_path=file_path,
        )

    def load_skill(self, skill_id: str) -> Skill:
        """
        【渐进加载核心接口】按需完整加载Skill实例
        存在缓存直接返回；无缓存读取磁盘解析完整内容
        """
        # 1. 校验是否已注册
        if skill_id not in self._registry:
            raise KeyError(f"不存在该Skill，未注册: {skill_id}")
        # 2. 命中缓存，直接返回，不再读文件
        if skill_id in self.loaded_skill_cache:
            logger.debug(f"Skill {skill_id} 命中内存缓存，无需重新加载")
            return self.loaded_skill_cache[skill_id]
        # 3. 缓存未命中：读取文件，加载完整Skill内容
        logger.info(f"渐进加载Skill完整资源: {skill_id}")
        meta = self._registry[skill_id]
        full_skill = self._parse_full_skill(meta)
        # 4. 存入缓存
        self.loaded_skill_cache[skill_id] = full_skill
        return full_skill

    def _parse_full_skill(self, meta: SkillMeta) -> Skill:
        """读取完整md文件，解析Prompt、参数、代码块"""
        with open(meta.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取Params
        param_match = re.search(r"## Params\n(.*?)(?=\n## |$)",content, re.DOTALL)
        params_schema = yaml.safe_load(param_match.group(1)) if param_match else {}

        # 提取Prompt
        prompt_match = re.search(r"## Prompt\n(.*?)(?=\n## Code|$)",content, re.DOTALL)
        prompt_template = prompt_match.group(1).strip() if prompt_match else ""

        # 提取可选代码块
        code_match = re.search(r"## Code\n```python\n(.*?)```",content, re.DOTALL)
        code_block = code_match.group(1) if code_match else None

        return Skill(
            meta=meta,
            prompt_template=prompt_template,
            parms_schema=params_schema,
            code_block=code_block,
        )

    def execute(self, skill_id: str, inputs: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        对外统一执行入口（Agent调用这个方法！）
        :param skill_id: 需要执行的技能ID
        :param inputs: 调用入参
        :param session_context: 当前会话上下文（session、memory信息）
        :return: 标准化执行结果
        """
        try:
            # 渐进加载：需要执行时才加载完整skill
            skill = self.load_skill(skill_id)

            # 参数校验简易实现（作业可扩展json schema校验）
            for param_name in skill.parms_schema.keys():
                if param_name not in inputs:
                    raise ValueError(f"Skill {skill_id} 缺少必填参数: {param_name}")

            # ========== 此处预留两种执行分支 ==========
            # 分支1：纯提示词Skill：把prompt+参数送入LLM
            # 分支2：带代码Skill：动态执行内嵌code_block
            # 你后续对接agent.py、llm_config.py在这里拓展

            # 【占位逻辑，后续替换真实执行】
            result_data = {
                "skill_id": skill_id,
                "status": "success",
                "prompt": skill.prompt_template,
                "inputs": inputs,
                "context": session_context,
            }
            return result_data

        except Exception as e:
            logger.exception(f"Skill执行异常 skill_id={skill_id}")
            return {
                "skill_id": skill_id,
                "status": "failed",
                "error": str(e),
            }

    def unload_skill(self, skill_id: str):
        """可选：释放缓存，卸载skill（加分扩展）"""
        if skill_id in self.loaded_skill_cache:
            del self.loaded_skill_cache[skill_id]
            logger.info(f"已卸载Skill缓存: {skill_id}")

    def list_available_skills(self) -> List[Skill]:
        """供Agent查询当前有哪些可用技能（只返回元信息，轻量化）"""
        return list(self._registry.values())
