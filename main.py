import os
import json
from ortools.sat.python import cp_model
from collections import defaultdict


def solve_displib_instance(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    trains = data["trains"]
    objectives = data.get("objective", [])

    # ===== 构建操作数据 =====
    operations = []
    op_id = 0
    op_map = {}
    for t_idx, train in enumerate(trains):
        for o_idx, op in enumerate(train):
            operations.append({
                "id": op_id,
                "train": t_idx,
                "op_idx": o_idx,
                "min_duration": op["min_duration"],
                "start_lb": op.get("start_lb", 0),
                "start_ub": op.get("start_ub", 9999),
                "resources": [r["resource"] for r in op.get("resources", [])],
                "successors": op.get("successors", [])
            })
            op_map[(t_idx, o_idx)] = op_id
            op_id += 1

    model = cp_model.CpModel()
    horizon = 10000

    # ===== 创建变量 =====
    start_vars = {}
    end_vars = {}
    intervals = {}
    for op in operations:
        op_id = op["id"]
        s = model.NewIntVar(op["start_lb"], op["start_ub"], f"start_{op_id}")
        e = model.NewIntVar(0, horizon, f"end_{op_id}")
        d = op["min_duration"]
        interval = model.NewIntervalVar(s, d, e, f"interval_{op_id}")
        start_vars[op_id] = s
        end_vars[op_id] = e
        intervals[op_id] = interval

    # ===== successor 顺序约束（基础） =====
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

    # ===== successor 选择变量（多路径） =====
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
                if succ_train != t:
                    continue
                succ_id = op_map.get((succ_train, succ_idx))
                if succ_id is not None:
                    y_var = model.NewBoolVar(f"y_{t}_{j}_{succ_idx}")
                    successor_choice_vars[(t, j)][succ_idx] = y_var
                    y_vars.append(y_var)
                    model.Add(start_vars[succ_id] >= end_vars[op["id"]]).OnlyEnforceIf(y_var)

        if y_vars:
            model.Add(sum(y_vars) == 1)

    # ===== 添加互斥路径选择约束（路径冲突排他性） =====
    # 如果两个操作的 successor 是同一个资源区段（比如 r3 → r4），不能都选
    conflict_pairs = []

    # 可选：你也可以读取 JSON 的 conflict_pair 字段（如果定义了）
    # 这里只是静态构建一些冲突对（示意）
    for (t1, j1), succs1 in successor_choice_vars.items():
        for (t2, j2), succs2 in successor_choice_vars.items():
            if (t1, j1) >= (t2, j2):
                continue  # 避免重复
            for succ_idx1, y1 in succs1.items():
                for succ_idx2, y2 in succs2.items():
                    # 如果 successor 是同一资源（例如都在用 r3 → r4）
                    res1 = operations[op_map[(t1, j1)]]["resources"]
                    res2 = operations[op_map[(t2, j2)]]["resources"]
                    common = set(res1) & set(res2)
                    if common:
                        # 添加互斥约束
                        model.AddBoolOr([y1.Not(), y2.Not()])

    # ===== 路径完整性（连续性）约束 =====
    for op in operations:
        if op["op_idx"] == 0:
            continue
        prev_key = (op["train"], op["op_idx"] - 1)
        curr_key = (op["train"], op["op_idx"])
        if prev_key in op_map and curr_key in op_map:
            pred_id = op_map[prev_key]
            curr_id = op_map[curr_key]
            model.Add(start_vars[curr_id] >= end_vars[pred_id])

    # ===== 资源冲突 + headway =====
    headway = 3
    resource_to_ops = defaultdict(list)
    for op in operations:
        for r in op["resources"]:
            resource_to_ops[r].append(op["id"])

    for res, ops in resource_to_ops.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a = ops[i]
                b = ops[j]
                bvar = model.NewBoolVar(f"order_{a}_{b}")
                model.Add(start_vars[a] + operations[a]["min_duration"] + headway <= start_vars[b]).OnlyEnforceIf(bvar)
                model.Add(start_vars[b] + operations[b]["min_duration"] + headway <= start_vars[a]).OnlyEnforceIf(bvar.Not())

    # ===== 资源释放时间约束（release_time） =====
    for res, op_ids in resource_to_ops.items():
        for i in range(len(op_ids)):
            for j in range(i + 1, len(op_ids)):
                a = op_ids[i]
                b = op_ids[j]

                release_a = 0
                release_b = 0

                for r_idx, r in enumerate(operations[a]["resources"]):
                    if r == res and "release_time" in data["trains"][operations[a]["train"]][operations[a]["op_idx"]]["resources"][r_idx]:
                        release_a = data["trains"][operations[a]["train"]][operations[a]["op_idx"]]["resources"][r_idx]["release_time"]
                for r_idx, r in enumerate(operations[b]["resources"]):
                    if r == res and "release_time" in data["trains"][operations[b]["train"]][operations[b]["op_idx"]]["resources"][r_idx]:
                        release_b = data["trains"][operations[b]["train"]][operations[b]["op_idx"]]["resources"][r_idx]["release_time"]

                bvar = model.NewBoolVar(f"conflict_{a}_{b}_res_{res}")
                model.Add(start_vars[a] + operations[a]["min_duration"] + release_a <= start_vars[b]).OnlyEnforceIf(bvar)
                model.Add(start_vars[b] + operations[b]["min_duration"] + release_b <= start_vars[a]).OnlyEnforceIf(bvar.Not())

    # ===== 站台容量约束（禁止资源重叠） =====
    for res, op_ids in resource_to_ops.items():
        res_intervals = [intervals[op_id] for op_id in op_ids]
        model.AddNoOverlap(res_intervals)

    # ===== 目标函数：op_delay + increment 支持 =====
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
