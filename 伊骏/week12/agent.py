"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent
新增：多轮对话交互式聊天模式
使用方式：
# 1. 原有单次执行（兼容旧代码）
python agent.py --mode manual --question "茅台2023年毛利率是多少？"
# 2. 交互式多轮对话（不带--question）
python agent.py --mode manual
python agent.py --mode fc
聊天指令：
exit    退出对话
clear   清空全部对话历史
history 打印当前所有对话记录
环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen-max，可换 deepseek-v3 等
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

def interactive_chat(run_print_func, max_steps):
    """交互式多轮对话主循环"""
    # 全局对话记忆：存储每一轮 (用户问题, AI最终回答)
    chat_history = []
    print("===== 金融Agent多轮对话模式 =====")
    print("指令：exit退出 | clear清空历史 | history查看对话\n")

    while True:
        user_input = input("你：").strip()
        # 处理内置指令
        if user_input.lower() == "exit":
            print("对话结束，再见！")
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("✅ 对话历史已清空\n")
            continue
        if user_input.lower() == "history":
            print("===== 当前对话历史 =====")
            for idx, (q,a) in enumerate(chat_history, 1):
                print(f"【第{idx}轮】用户：{q}")
                print(f"【第{idx}轮】Agent：{a}\n")
            continue
        if not user_input:
            print("请输入有效问题\n")
            continue

        # 调用底层react，传入历史对话，获取本轮回答
        final_answer = run_print_func(user_input, max_steps, chat_history=chat_history)
        # 把本轮问答存入记忆，下一轮自动携带上下文
        chat_history.append((user_input, final_answer))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent（支持多轮对话）")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=None, help="单次提问；不传则进入交互式多轮对话")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    # 根据模式导入对应run_and_print
    if args.mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print
    # 分支1：传入--question，执行原有单次问答（兼容旧代码）
    if args.question is not None:
        run_and_print(args.question, args.max_steps)
    # 分支2：未传--question，启动交互式多轮对话
    else:
        interactive_chat(run_and_print, args.max_steps)
