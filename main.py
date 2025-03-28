import json
from ortools.sat.python import cp_model
from collections import defaultdict

def solve_displib_instance(json_path):
    # 读取JSON数据
    with open(json_path, "r") as f:
        data = json.load(f)

    trains = data["trains"]
    objectives = data.get("objective", [])


    # ===== 构建全局操作列表 =====
    operations = []
    op_id = 0
    op_map = {}  # 映射 (train_idx, op_idx) -> global op id
    for t_idx, train in enumerate(trains):
        for o_idx, op in enumerate(train):
            operations.append({
                "id": op_id,
                "train": t_idx,
                "op_idx": o_idx,
                "min_duration": op["min_duration"],
                "start_lb": op.get("start_lb", 0),
                "start_ub": op.get("start_ub", 9999),
                # resources字段：取每个资源的名称，release_time在JSON中单独定义（如果有）
                "resources": [r["resource"] for r in op.get("resources", [])],
                "release_times": [r.get("release_time", 0) for r in op.get("resources", [])],
                "successors": op.get("successors", [])
            })
            op_map[(t_idx, o_idx)] = op_id
            op_id += 1

    model = cp_model.CpModel()
    horizon = 10000  # 时间上界

    # ===== 创建变量 =====
    start_vars = {}   # 操作开始时间变量
    end_vars = {}     # 操作结束时间变量
    intervals = {}    # 用于资源不重叠约束的区间变量
    for op in operations:
        oid = op["id"]
        s = model.NewIntVar(op["start_lb"], op["start_ub"], f"start_{oid}")
        e = model.NewIntVar(0, horizon, f"end_{oid}")
        d = op["min_duration"]
        interval = model.NewIntervalVar(s, d, e, f"interval_{oid}")
        start_vars[oid] = s
        end_vars[oid] = e
        intervals[oid] = interval

    # ===== 基础后继顺序约束 =====
    # 若Op2为Op1的后继，则要求： start(Op2) ≥ end(Op1)
    for op in operations:
        pred = op["id"]
        for succ in op["successors"]:
            if isinstance(succ, str) and '_' in succ:
                parts = succ[1:].split('_')
                train = int(parts[0])
                op_idx = int(parts[1])
                succ_id = op_map.get((train, op_idx))
                if succ_id is not None:
                    model.Add(start_vars[succ_id] >= end_vars[pred])

    # ===== 多路径选择变量（若有多个后继选项，仅选择一条） =====
    successor_choice_vars = defaultdict(dict)
    for op in operations:
        t = op["train"]
        j = op["op_idx"]
        succs = op["successors"]
        if not succs or len(succs) == 1:
            continue
        y_vars = []
        for succ in succs:
            if isinstance(succ, str) and "_" in succ:
                parts = succ[1:].split("_")
                succ_train, succ_idx = int(parts[0]), int(parts[1])
                # 只考虑同一列车内的多路径情况
                if succ_train != t:
                    continue
                succ_id = op_map.get((succ_train, succ_idx))
                if succ_id is not None:
                    y_var = model.NewBoolVar(f"y_{t}_{j}_{succ_idx}")
                    successor_choice_vars[(t, j)][succ_idx] = y_var
                    y_vars.append(y_var)
                    # 若选中该后继，则添加额外延时（extra_delay），用于调节路径选择产生的延迟
                    extra_delay = 5 # 这里可根据实际需要设置，比如对于swapping实例可设为非0值
                    model.Add(start_vars[succ_id] >= end_vars[op["id"]] + extra_delay).OnlyEnforceIf(y_var)
        if y_vars:
            model.Add(sum(y_vars) == 1)

    # ===== 互斥路径选择约束 =====
    # 如果两个操作的后继选择共享相同的资源，则不能同时选择
    for (t1, j1), succs1 in successor_choice_vars.items():
        for (t2, j2), succs2 in successor_choice_vars.items():
            if (t1, j1) >= (t2, j2):
                continue
            for succ_idx1, y1 in succs1.items():
                for succ_idx2, y2 in succs2.items():
                    res1 = operations[op_map[(t1, j1)]]["resources"]
                    res2 = operations[op_map[(t2, j2)]]["resources"]
                    common = set(res1) & set(res2)
                    if common:
                        model.AddBoolOr([y1.Not(), y2.Not()])

    # ===== 路径连续性约束 =====
    # 同一列车内连续操作必须依次执行
    for op in operations:
        if op["op_idx"] == 0:
            continue
        prev_key = (op["train"], op["op_idx"] - 1)
        curr_key = (op["train"], op["op_idx"])
        if prev_key in op_map and curr_key in op_map:
            pred_id = op_map[prev_key]
            curr_id = op_map[curr_key]
            model.Add(start_vars[curr_id] >= end_vars[pred_id])

    # ===== 资源冲突与Headway约束 =====
    headway = 3  # 安全间隔
    resource_to_ops = defaultdict(list)
    for op in operations:
        for r in op["resources"]:
            resource_to_ops[r].append(op["id"])

    for res, ops in resource_to_ops.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a = ops[i]
                b = ops[j]
                order_bool = model.NewBoolVar(f"order_{a}_{b}")
                model.Add(start_vars[a] + operations[a]["min_duration"] + headway <= start_vars[b]).OnlyEnforceIf(order_bool)
                model.Add(start_vars[b] + operations[b]["min_duration"] + headway <= start_vars[a]).OnlyEnforceIf(order_bool.Not())

    # ===== 资源释放时间约束（Release time） =====
    # 考虑每个操作在释放资源后，资源才可供下一个操作使用的延时
    for res, op_ids in resource_to_ops.items():
        for i in range(len(op_ids)):
            for j in range(i + 1, len(op_ids)):
                a = op_ids[i]
                b = op_ids[j]
                # 默认释放时间为0，如JSON中定义了则使用其值
                release_a = operations[a]["release_times"][operations[a]["resources"].index(res)] if res in operations[a]["resources"] else 0
                release_b = operations[b]["release_times"][operations[b]["resources"].index(res)] if res in operations[b]["resources"] else 0
                rel_bool = model.NewBoolVar(f"release_conflict_{a}_{b}_res_{res}")
                model.Add(start_vars[a] + operations[a]["min_duration"] + release_a <= start_vars[b]).OnlyEnforceIf(rel_bool)
                model.Add(start_vars[b] + operations[b]["min_duration"] + release_b <= start_vars[a]).OnlyEnforceIf(rel_bool.Not())

    # ===== 站台容量约束（No overlap） =====
    for res, op_ids in resource_to_ops.items():
        res_intervals = [intervals[oid] for oid in op_ids]
        model.AddNoOverlap(res_intervals)

    # ===== 目标函数：最小化关键操作延迟（Delay penalty + increment） =====
    penalties = []
    for obj in objectives:
        if obj["type"] == "op_delay":
            key = (obj["train"], obj["operation"])
            if key not in op_map:
                continue
            op_id = op_map[key]
            start = start_vars[op_id]
            threshold = obj.get("threshold", 0)
            coeff = obj.get("coeff", 1)
            increment = obj.get("increment", 0)
            # 计算延迟 = max(0, start - threshold)
            delay = model.NewIntVar(0, horizon, f"delay_{op_id}")
            model.Add(delay >= start - threshold)
            model.Add(delay >= 0)
            if increment > 0:
                has_delay = model.NewBoolVar(f"has_delay_{op_id}")
                model.Add(delay > 0).OnlyEnforceIf(has_delay)
                model.Add(delay == 0).OnlyEnforceIf(has_delay.Not())
                inc_var = model.NewIntVar(0, increment, f"inc_{op_id}")
                model.Add(inc_var == increment).OnlyEnforceIf(has_delay)
                model.Add(inc_var == 0).OnlyEnforceIf(has_delay.Not())
                penalty = model.NewIntVar(0, horizon * coeff + increment, f"penalty_{op_id}")
                model.Add(penalty == delay * coeff + inc_var)
            else:
                penalty = model.NewIntVar(0, horizon * coeff, f"penalty_{op_id}")
                model.AddMultiplicationEquality(penalty, [delay, coeff])
            penalties.append(penalty)

    if penalties:
        total_penalty = model.NewIntVar(0, horizon * len(penalties), "total_penalty")
        model.Add(total_penalty == sum(penalties))
        model.Minimize(total_penalty)
    else:
        total_penalty = None

    # ===== 求解模型 =====
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # ===== 输出结果 =====
    results = {"events": [], "objective_value": None}
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for op in operations:
            results["events"].append({
                "operation": op["op_idx"],
                "train": op["train"],
                "time": solver.Value(start_vars[op["id"]])
            })
        results["events"].sort(key=lambda x: x["time"])
        results["objective_value"] = solver.Value(total_penalty) if total_penalty is not None else 0

    return results


# ===== 主程序入口 =====
if __name__ == "__main__":
    for name in ["headway1", "swapping1", "swapping2", "infeasible1", "infeasible2"]:
        path = f"/Users/wendyli/Downloads/displib_instances_testing/displib_instances_testing/displib_testinstances_{name}.json"
        print(f"\n📄 Solving: {name}")
        result = solve_displib_instance(path)
        print(json.dumps(result, indent=2))


import os
import json

# 设置你的 JSON 文件路径
json_path = "/Users/wendyli/Downloads/displib_instances_testing/displib_instances_testing/displib_testinstances_swapping1.json"

with open(json_path, "r") as f:
    data = json.load(f)

trains = data["trains"]

for t_idx, train in enumerate(trains):
    for o_idx, op in enumerate(train):
        resources = op.get("resources", [])
        for r in resources:
            # 如果没有定义 release_time，则 r 可能只有 "resource" 字段
            if "release_time" not in r:
                print(f"Train {t_idx}, Operation {o_idx} 的资源 {r['resource']} 没有定义 release_time")
            else:
                print(f"Train {t_idx}, Operation {o_idx} 的资源 {r['resource']} 的 release_time = {r['release_time']}")
