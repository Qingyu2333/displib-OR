import gurobipy as gp
from gurobipy import GRB
import json
import os
from READ_BUILD_MODEL import read_displib_json, build_mip_model
import pandas as pd
from collections import OrderedDict

def extract_gurobi_stats(model, label="Default"):
    stats = {
        "Label": label,
        "Runtime (s)": round(model.Runtime, 2),
        "Node Count": model.NodeCount,
        "Simplex Iterations": model.IterCount,
        "Best Objective": round(model.ObjVal, 4),
        "Best Bound": round(model.ObjBound, 4),
        "MIP Gap (%)": round(model.MIPGap * 100, 2)
    }

    print(f"\n📊 [{label}] Gurobi 求解性能统计")
    for key, val in stats.items():
        print(f"{key:<20}: {val}")
    
    return stats




if __name__ == "__main__":
    # 读取 JSON 数据
    #filepath = "C:\\Users\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_testing\\displib_instances_testing\\displib_testinstances_infeasible1.json"
    #filepath = "C:\\Users\\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_phase1_v1_1\\displib_instances_phase1\\line3_1.json"
    filepath = "C:\\Users\\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_testing\\displib_instances_testing\\displib_testinstances_headway1.json"
    displib_data = read_displib_json(filepath)

    print("✅ JSON读取成功！")

    trains = displib_data['trains']
    operations = displib_data['operations']
    conflict_pairs = displib_data['conflict_pairs']
    train_paths = displib_data['train_paths']
    headways = displib_data['headways']
    time_windows = displib_data['time_windows']
    objectives = displib_data['objectives']

    model, t, active, y = build_mip_model(
        trains, operations, conflict_pairs, train_paths, headways, time_windows, objectives
    )

    model.setParam('MIPGap', 0.001)
    model.setParam('OptimalityTol', 1e-9)
    model.setParam('FeasibilityTol', 1e-9)
    model.optimize()

    label = "With Cutting Planes"
    stats = extract_gurobi_stats(model, label=label)
##    output_csv = r"C:\\Users\\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_phase1_v1_1\\stats\\critical_heuristics1.csv"
##    if os.path.exists(output_csv):
##        df_prev = pd.read_csv(output_csv)
##        df_new = pd.concat([df_prev, pd.DataFrame([stats])], ignore_index=True)
##    else:
##        df_new = pd.DataFrame([stats])
##    df_new.to_csv(output_csv, index=False)
##    print(f"✅ 求解统计结果写入：{output_csv}")

    if model.status == GRB.OPTIMAL:
        print("✅ 最优解找到！")

        def get_op_end_time(train_id, op_id, start_time):
            for op in operations:
                if op['train'] == train_id and op['op_idx'] == op_id:
                    dur = op['min_duration']
                    release = max(op['resource_release_times']) if op['resource_release_times'] else 0
                    return start_time + dur + release
            return start_time

        solution = {
            "objective_value": round(model.ObjVal, 6),
            "events": []
        }

        for train_idx in range(len(trains)):
            ops_in_train = [op for op in operations if op['train'] == train_idx]
            start_ops = [op for op in ops_in_train if not op.get('predecessors')]
            if not start_ops:
                continue
            current_op = start_ops[0]['op_idx']
            while True:
                if active[train_idx, current_op].X < 0.5 or t[train_idx, current_op].X >= 1e6:
                    break
                solution["events"].append({
                    "train": train_idx,
                    "operation": current_op,
                    "time": int(round(t[train_idx, current_op].X))
                })
                next_op = None
                for s in [op['op_idx'] for op in operations if op['train'] == train_idx]:
                    if (train_idx, current_op, s) in y.keys() and y[train_idx, current_op, s].X > 0.5:
                        next_op = s
                        break
                if next_op is None:
                    break
                current_op = next_op

        # Swapping
        def get_release_end(event):
            for op in operations:
                if op['train'] == event['train'] and op['op_idx'] == event['operation']:
                    release = max(op['resource_release_times']) if op['resource_release_times'] else 0
                    return event['time'] + op['min_duration'] + release
            return event['time']

        solution["events"] = sorted(
            solution["events"],
            key=lambda x: (x['time'], x['operation'], x['train'])
        )
        solution["events"] = [
            OrderedDict([
                ("operation", e["operation"]),
                ("time", e["time"]),
                ("train", e["train"])
            ]) for e in solution["events"]
        ]

        output_dir = r"C:\\Users\\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_phase1_v1_1\\solution"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "headway.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(solution, f, indent=4, ensure_ascii=False)

        print(f"✅ 解决方案已保存：{output_path}")

    else:
        print("❌ 没有找到最优解！")

