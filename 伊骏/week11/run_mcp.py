"""
run_mcp.py — 改造版：支持多轮循环工具调用（原单轮闭环升级）
修复NameError传参bug；新增while循环迭代执行工具、最大调用轮次限制防死循环
"""
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

# ── LLM 配置（与原版保持一致）──────────────
PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── Server 配置 ────────────────────────────────────────────────────────────
def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "rag": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "rag_server.py")],
            env={**os.environ},
        ),
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


# ── 连接所有 Server：一次走完 建管道→握手→发现工具→转 schema ───────────────
async def connect_all_servers(stack: AsyncExitStack):
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []
    for label, params in build_server_configs().items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        print(f"  ✓ [{label}]  {', '.join(t.name for t in tools_result.tools)}", file=sys.stderr)
    print(f"\n共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


# ── 系统提示词不变 ───────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。本回合你可以一次调用多个工具，也可以分多轮多次调用工具。"
)


# ===================== 改造核心：支持多轮循环工具调用 =====================
async def run(client, model: str, question: str,
              tool_registry: dict, openai_tools: list[dict], verbose: bool = True,
              max_loop_round: int = 5) -> dict:
    """
    多轮循环闭环：持续调用工具直到LLM无需工具
    :param max_loop_round: 最大循环轮次，防止无限递归调用工具
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    loop_count = 0
    # 循环：只要还有工具调用且未达最大轮次，就持续执行
    while loop_count < max_loop_round:
        loop_count += 1
        if verbose:
            print(f"\n===== 工具调用循环第 {loop_count} 轮 =====", file=sys.stderr)
        # 请求LLM生成回复/工具调用
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message
        # 无工具调用，直接退出循环
        if not msg.tool_calls:
            break
        # 存在工具调用，执行全部工具
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "loop_round": loop_count})
            if verbose:
                print(f"  → [mcp] {name}({args})")
            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # 循环结束，生成最终总结回答
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=openai_tools, tool_choice="auto",
    )
    msg = resp.choices[0].message
    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"\n  → [llm] 最终回答（总耗时 {elapsed:.1f}s，总循环轮次 {loop_count}）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "total_loop_round": loop_count
    }


# ── 入口DEMO问题 ───────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",
    # 新增测试多轮循环工具调用的问题
    "先查宁德天气，再查深圳天气，最后查询厦门当前温度，汇总三地温差对比",
]

# 修复点1：main_async增加verbose入参，不再内部读取args
async def main_async(provider: str, question: str | None, demo: bool, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)
    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)
        questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        results = []
        for i, q in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            # 修复点2：直接传入参verbose，不再写not args.quiet
            result = await run(client, model, q, tool_registry, openai_tools, verbose=verbose)
            result["question"] = q
            results.append(result)
            if not as_json:
                print("\n最终回答：")
                print(result["answer"])
                print()
        if as_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP（支持多轮循环工具调用）")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()
    # 修复点3：提前计算verbose布尔值，传入main_async
    verbose_flag = not args.quiet
    asyncio.run(main_async(
        provider=args.provider,
        question=args.question,
        demo=args.demo,
        verbose=verbose_flag,
        as_json=args.json
    ))


if __name__ == "__main__":
    main()
