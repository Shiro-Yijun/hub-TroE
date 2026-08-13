## Meta
id: attraction_planning
name: 旅游并行子Agent调度器
tags: ["multi-agent", "parallel", "travel", "任务拆分"]
description: 多线程并行执行景点路线、食宿预算双SubAgent，自动整合完整旅游方案
file_path: ./skills/attraction_planning.md
## Params
tasks: list[str]
city: str
travel_days: int
people_count: int
budget_level: str
summary_prompt: str
## Prompt
仅占位文本，本技能优先执行内置Python并行代码
## Code
```python
def run(params, ctx):  #其实这个就是主agent
    # 全部导入放函数内部，解决exec作用域丢失问题
    import concurrent.futures
    import logging
    import time
    from src.llm_config import get_chat_client
    sub_logger = logging.getLogger("sub_agent_travel")

    def llm_retry_call(client, model, messages, temperature, timeout, max_retry=2):
        """LLM 调用重试封装"""
        err = None
        for i in range(max_retry):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout
                )
                return resp
            except Exception as e:
                err = e
                sub_logger.warning(f"LLM调用失败，重试{i+1}/{max_retry}: {str(e)}")
                time.sleep(1)
        raise RuntimeError(f"LLM多次调用失败: {str(err)}")

    def run_single_subtask(sub_task, global_ctx, travel_params, task_rule_map): #这个就是sub agent
        sub_logger.info(f"启动子任务：{sub_task}")
        try:
            # 全局资源校验
            required_res = ["db", "loader", "vs", "fts", "retriever"]
            for res_key in required_res:
                if res_key not in global_ctx:
                    raise RuntimeError(f"全局资源缺失: {res_key}")
            db = global_ctx["db"]
            loader = global_ctx["loader"]
            vs = global_ctx["vs"]
            fts = global_ctx["fts"]
            retriever = global_ctx["retriever"]

            sid = db.new_session()
            prompt_data = loader.build_system_prompt(recent_memory_limit=3)
            base_prompt = prompt_data.system_prompt[:2500]
            search_text = f"{travel_params['city']} {sub_task}"
            search_res = retriever.search(search_text, top_k=2)
            mem_lines = [f"-{item['content'][:100]}" for item in search_res] if search_res else []
            mem_block = "\n".join(mem_lines)
            rule = task_rule_map[sub_task]
            full_sys_prompt = f"""
你是旅游规划系统中的“{sub_task}”专业 SubAgent。
必须严格只完成当前子任务，不要替另一个 SubAgent 工作。
{rule}
出行参数：
{travel_params}
历史参考记忆：
{mem_block or "无"}
系统通用约束（仅作背景参考）：
{base_prompt}
"""
            client, model = get_chat_client()
            resp = llm_retry_call(
                client=client,
                model=model,
                messages=[{"role":"system","content":full_sys_prompt},{"role":"user","content":"输出完整结构化内容"}],
                temperature=0.6,
                timeout=60
            )
            output_text = resp.choices[0].message.content.strip()
            db.add_message(sid, "user", sub_task)
            db.add_message(sid, "assistant", output_text)
            db.close_session(sid)
            return {"sub_task_name": sub_task, "task_output": output_text, "status": "success"}
        except Exception as err:
            sub_logger.exception(f"子任务 {sub_task} 执行异常")
            return {"sub_task_name": sub_task, "task_output": f"子任务异常：{str(err)}", "status": "fail"}

    # ========== run函数主逻辑 ==========
    # 全局资源容错
    if "global_resource" not in ctx:
        return {
            "run_status": "failed",
            "error": "会话上下文缺失 global_resource 全局资源",
            "sub_task_list": [],
            "scenic_raw": "",
            "budget_raw": "",
            "final_travel_plan": ""
        }
    global_resources = ctx["global_resource"]
    task_list = params["tasks"]
    if isinstance(task_list, str):
        try:
            import json
            task_list = json.loads(task_list)
        except Exception:
            task_list = [task_list]
    if not isinstance(task_list, list):
        raise ValueError("tasks 必须是列表")
    required_tasks = ["景点路线规划", "食宿预算核算"]
    missing_tasks = [t for t in required_tasks if t not in task_list]
    if missing_tasks:
        raise ValueError(f"tasks 缺少必需子任务: {missing_tasks}")
    task_list = required_tasks
    travel_args = {
        "city": params["city"],
        "travel_days": int(params["travel_days"]),
        "people_count": int(params["people_count"]),
        "budget_level": params["budget_level"]
    }
    summary_text = params["summary_prompt"]
    task_rules = {
        "景点路线规划": "仅输出每日景点、位置、交通、游览时长、景点介绍，禁止任何价格、住宿、餐饮文字",
        "食宿预算核算": "仅输出住宿、餐饮、门票分项费用，禁止任何行程、景点描述文字"
    }
    task_results_map = {}   # 并发分发，主 Agent 调用多个 SubAgent 的关键代码段
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
            task_futures = {}
            for t in task_list:
                fut = exe.submit(run_single_subtask, t, global_resources, travel_args, task_rules)
                task_futures[fut] = t
            done, not_done = concurrent.futures.wait(task_futures.keys(), timeout=260)
            for fut in not_done:
                t_name = task_futures[fut]
                task_results_map[t_name] = {
                    "sub_task_name": t_name,
                    "task_output": "子任务执行超时",
                    "status": "fail"
                }
            for fut in done:
                t_name = task_futures[fut]
                task_results_map[t_name] = fut.result()
    except Exception as e:
        sub_logger.exception("线程池并行执行异常")
        return {
            "run_status": "thread_pool_error",
            "error": str(e),
            "sub_task_list": [],
            "scenic_raw": "",
            "budget_raw": "",
            "final_travel_plan": ""
        }
    scenic_data = task_results_map.get("景点路线规划", {"task_output":"未生成景点路线","status":"fail"})
    budget_data = task_results_map.get("食宿预算核算", {"task_output":"未生成预算","status":"fail"})
    merge_prompt = f"""
{summary_text}
===== 景点路线原始内容 =====
{scenic_data['task_output']}
===== 食宿预算原始内容 =====
{budget_data['task_output']}
要求：按天数拆分，每一天先放游玩路线，再放当日对应花费，结构清晰完整。
"""
    client, model = get_chat_client()
    merge_resp = llm_retry_call(
        client=client,
        model=model,
        messages=[{"role":"user","content":merge_prompt}],
        temperature=0.7,
        timeout=60
    )
    full_plan = merge_resp.choices[0].message.content.strip()
    return {
        "run_status": "all_finished",
        "sub_task_list": list(task_results_map.values()),
        "scenic_raw": scenic_data["task_output"],
        "budget_raw": budget_data["task_output"],
        "final_travel_plan": full_plan
    }
```