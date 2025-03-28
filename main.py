import os
import json
from ortools.sat.python import cp_model
from collections import defaultdict

def add_path_segment_conflict_intervals(model, operations, start_vars):
    segment_to_intervals = defaultdict(list)
    for op in operations:
        res = op["resources"]
        if len(res) >= 2:
            for i in range(len(res) - 1):
                segment = (res[i], res[i + 1])
                interval = model.NewIntervalVar(
                    start_vars[op["id"]],
                    op["min_duration"],
                    start_vars[op["id"]] + op["min_duration"],
                    f"pathseg_interval_train{op['train']}_op{op['op_idx']}_{i}"
                )
                segment_to_intervals[segment].append(interval)
    for segment, intervals in segment_to_intervals.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

def solve_displib_instance(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    trains = data["trains"]
    objectives = data.get("objective", [])

    # Build global operation list
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
                "release_times": [r.get("release_time", 0) for r in op.get("resources", [])],
                "successors": op.get("successors", [])
            })
            op_map[(t_idx, o_idx)] = op_id
            op_id += 1

    model = cp_model.CpModel()
    horizon = 10000

    # Create variables: start, end, and interval for each operation
    start_vars = {}
    end_vars = {}
    intervals = {}
    for op in operations:
        oid = op["id"]
        s = model.NewIntVar(op["start_lb"], op["start_ub"], f"start_{oid}")
        e = model.NewIntVar(0, horizon, f"end_{oid}")
        d = op["min_duration"]
        interval = model.NewIntervalVar(s, d, e, f"interval_{oid}")
        start_vars[oid] = s
        end_vars[oid] = e
        intervals[oid] = interval

    # Precedence constraints
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

    # Path selection logic
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
                    extra_delay = 5
                    model.Add(start_vars[succ_id] >= end_vars[op["id"]] + extra_delay).OnlyEnforceIf(y_var)
        if y_vars:
            model.Add(sum(y_vars) == 1)

    # Mutual exclusion for selected successor paths
    for (t1, j1), succs1 in successor_choice_vars.items():
        for (t2, j2), succs2 in successor_choice_vars.items():
            if (t1, j1) >= (t2, j2):
                continue
            for succ_idx1, y1 in succs1.items():
                for succ_idx2, y2 in succs2.items():
                    res1 = operations[op_map[(t1, j1)]]["resources"]
                    res2 = operations[op_map[(t2, j2)]]["resources"]
                    if set(res1) & set(res2):
                        model.AddBoolOr([y1.Not(), y2.Not()])

    # Train internal path continuity
    for op in operations:
        if op["op_idx"] == 0:
            continue
        prev_key = (op["train"], op["op_idx"] - 1)
        curr_key = (op["train"], op["op_idx"])
        if prev_key in op_map and curr_key in op_map:
            model.Add(start_vars[op_map[curr_key]] >= end_vars[op_map[prev_key]])

    # Resource conflict with headway
    headway = 0
    resource_to_ops = defaultdict(list)
    for op in operations:
        for r in op["resources"]:
            resource_to_ops[r].append(op["id"])

    for res, ops in resource_to_ops.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a, b = ops[i], ops[j]
                bvar = model.NewBoolVar(f"order_{a}_{b}")
                model.Add(start_vars[a] + operations[a]["min_duration"] + headway <= start_vars[b]).OnlyEnforceIf(bvar)
                model.Add(start_vars[b] + operations[b]["min_duration"] + headway <= start_vars[a]).OnlyEnforceIf(bvar.Not())

    # No-overlap constraints
    for res, op_ids in resource_to_ops.items():
        model.AddNoOverlap([intervals[oid] for oid in op_ids])

    # Release time constraints
    for res, op_ids in resource_to_ops.items():
        for i in range(len(op_ids)):
            for j in range(i + 1, len(op_ids)):
                a, b = op_ids[i], op_ids[j]
                release_a = operations[a]["release_times"][operations[a]["resources"].index(res)] if res in operations[a]["resources"] else 0
                release_b = operations[b]["release_times"][operations[b]["resources"].index(res)] if res in operations[b]["resources"] else 0
                rel_bool = model.NewBoolVar(f"release_conflict_{a}_{b}_res_{res}")
                model.Add(start_vars[a] + operations[a]["min_duration"] + release_a <= start_vars[b]).OnlyEnforceIf(rel_bool)
                model.Add(start_vars[b] + operations[b]["min_duration"] + release_b <= start_vars[a]).OnlyEnforceIf(rel_bool.Not())

    # Segment conflict constraint
    add_path_segment_conflict_intervals(model, operations, start_vars)

    # Global priority constraint: train 0 op1 must precede train 1 op1
    buffer = 5
    if (0, 1) in op_map and (1, 1) in op_map:
        model.Add(start_vars[op_map[(1, 1)]] >= end_vars[op_map[(0, 1)]] + buffer)

    # Objective function: delay penalty
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

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    print(f"⏱ Solver wall time: {solver.WallTime():.3f} seconds")

    results = {"events": [], "objective_value": None}
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for op in operations:
            results["events"].append({
                "operation": op["op_idx"],
                "train": op["train"],
                "time": solver.Value(start_vars[op["id"]])
            })
        results["events"].sort(key=lambda x: x["time"])
        results["objective_value"] = solver.Value(total_penalty)

    return results


# Main entry point
if __name__ == "__main__":
    for name in ["headway1", "swapping1", "swapping2", "infeasible1", "infeasible2"]:
        path = f"/Users/wendyli/Downloads/displib_instances_testing/displib_instances_testing/displib_testinstances_{name}.json"
        print(f"\n📄 Solving: {name}")
        result = solve_displib_instance(path)
        print(json.dumps(result, indent=2))

