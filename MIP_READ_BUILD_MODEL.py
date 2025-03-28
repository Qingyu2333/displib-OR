import json

def read_displib_json(filepath):
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)

    trains = []
    operations = []
    resources = set()
    time_windows = []
    headways = []
    objectives = []

    resource_usage = {}    # Storage resource -> (train, op) list
    train_paths = {}       # Storage train -> [(op_idx, resource)]
    conflict_pairs = []    # Generate conflict pairs

    for train_idx, train_ops in enumerate(data.get('trains', [])):
        train_name = f"train_{train_idx}"
        trains.append(train_name)
        for op_idx, op in enumerate(train_ops):
            op_dict = {
                'train': train_idx,
                'op_idx': op_idx,
                'start_lb': op.get('start_lb', 0),
                'start_ub': op.get('start_ub', float('inf')),
                'min_duration': op['min_duration'],
                'successors': op.get('successors', []),
                'resources': [],
                'resource_release_times': []
            }

            ## Collect resources and release time
            for r in op.get('resources', []):
                res_name = r['resource']
                resources.add(res_name)
                resource_usage.setdefault(res_name, []).append((train_idx, op_idx))
                train_paths.setdefault(train_idx, []).append((op_idx, res_name))
                op_dict['resources'].append(res_name)
                op_dict['resource_release_times'].append(r.get('release_time', 0))
            assert op_dict['train'] == train_idx, f"[❌] Train mismatch: got {op_dict['train']} but expected {train_idx}"

            operations.append(op_dict)

            # collect time window
            time_windows.append({
                'train': train_idx,
                'op_idx': op_idx,
                'start_lb': op_dict['start_lb'],
                'start_ub': op_dict['start_ub']
            })

    # Conflict Pair
    for res, op_list in resource_usage.items():
        for idx1 in range(len(op_list)):
            for idx2 in range(idx1 + 1, len(op_list)):
                (i, j), (k, l) = op_list[idx1], op_list[idx2]
                if (i, j) == (k, l):
                    continue
                if i == k:
                    continue  # 同车内不冲突，path解决
                conflict_pairs.append(((i, j), (k, l), res))  # 正向
                conflict_pairs.append(((k, l), (i, j), res))  # 🔥 反向补上！



    # read objective
    for obj in data.get('objective', []):
        objectives.append({
            'type': obj.get('type'),
            'train': obj.get('train'),
            'operation': obj.get('operation'),
            'threshold': obj.get('threshold', 0),
            'increment': obj.get('increment', 0),
            'coeff': obj.get('coeff', 0)
        })

    # 读取 headways（如果有）
    if 'headways' in data:
        headways = data.get('headways', [])

    # ============ 反向生成 predecessors ==============
    for op in operations:
        op['predecessors'] = []

    for op in operations:
        i, j = op['train'], op['op_idx']
        for succ in op['successors']:
            for target_op in operations:
                if target_op['train'] == i and target_op['op_idx'] == succ:
                    target_op['predecessors'].append((i, j))
                    break

    return {
        'trains': trains,
        'operations': operations,
        'resources': list(resources),
        'time_windows': time_windows,
        'headways': headways,
        'objectives': objectives,
        'conflict_pairs': conflict_pairs,
        'train_paths': train_paths
    }

# run information
if __name__ == "__main__":
    filepath = ("C:\\Users\\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_phase1_v1_1\\displib_instances_phase1\\line3_1.json")
    displib_data = read_displib_json(filepath)
    print("读取成功！列车数：", len(displib_data['trains']))
    #print("The third operation示例：", displib_data['operations'][70])
    print("资源集合示例：", displib_data['resources'][:5])
    print("目标函数条目示例：", displib_data['objectives'][:2])
    print(displib_data['headways'])
    print(f"Number of headway constraints: {len(displib_data['headways'])}")
    operations = displib_data['operations']  
##    train_id = 0
##    ops_for_train0 = [op for op in operations if op['train'] == train_id]
##    # 打印结果
##    for op in ops_for_train0:
##        print(op)
if __name__ == "__main__":
    filepath = ("C:\\Users\\陆柯言\\Desktop\\大四第二学期学习资料\\应用运筹project\\displib_instances_phase1_v1_1\\displib_instances_phase1\\line3_1.json")
    displib_data = read_displib_json(filepath)

    print("✅ JSON读取成功！")
    print("场景规模如下：")
    print(f"- 列车数（Trains）：{len(displib_data['trains'])}")
    print(f"- 操作数（Operations）：{len(displib_data['operations'])}")
    print(f"- 资源数（Resources）：{len(displib_data['resources'])}")
    print(f"- 冲突对数（Conflict Pairs）：{len(displib_data['conflict_pairs'])}")
    print(f"- Headway 约束数：{len(displib_data['headways'])}")
    print(f"- Time window 条目数：{len(displib_data['time_windows'])}")
    print(f"- 目标函数组件数（Objectives）：{len(displib_data['objectives'])}")



import gurobipy as gp
from gurobipy import GRB

def build_mip_model(trains, operations, conflict_pairs, train_paths, headways, time_windows, objectives): 
    model = gp.Model("Train_Scheduling")

    # 定义决策变量
    op_keys = [(op['train'], op['op_idx']) for op in operations]
    t = model.addVars(op_keys, vtype=GRB.CONTINUOUS, name="t")
    b_keys = list({(i, j, k, l) for ((i, j), (k, l), _) in conflict_pairs})
    b = model.addVars(b_keys, vtype=GRB.BINARY, name="b")
    y = model.addVars([(op['train'], op['op_idx'], s) for op in operations for s in op['successors']], vtype=GRB.BINARY, name='y')
    active = model.addVars([(op['train'], op['op_idx']) for op in operations], vtype=GRB.BINARY, name="active")

    M = 1e6

    # ========================  Resource Conflict Constraints ========================
    for (i, j), (k, l), res in conflict_pairs:
        op_ij = next(op for op in operations if op['train'] == i and op['op_idx'] == j)
        op_kl = next(op for op in operations if op['train'] == k and op['op_idx'] == l)
        min_duration_ij = op_ij['min_duration']
        min_duration_kl = op_kl['min_duration']
        release_time_ij = 0
        if res in op_ij['resources']:
            idx = op_ij['resources'].index(res)
            release_time_ij = op_ij['resource_release_times'][idx]

        release_time_kl = 0
        if res in op_kl['resources']:
            idx = op_kl['resources'].index(res)
            release_time_kl = op_kl['resource_release_times'][idx]

        model.addConstr(b[i, j, k, l] + b[k, l, i, j] == 1, name=f"mutual_{i}_{j}_{k}_{l}")

        model.addConstr(
            t[i, j] + min_duration_ij + release_time_ij
            <= t[k, l] + M * (1 - b[i, j, k, l]) + M * (2 - active[i, j] - active[k, l]),
            name=f"conflict1_{i}_{j}_{k}_{l}"
        )

        model.addConstr(
            t[k, l] + min_duration_kl + release_time_kl
            <= t[i, j] + M * (1 - b[k, l, i, j]) + M * (2 - active[i, j] - active[k, l]),
            name=f"conflict2_{i}_{j}_{k}_{l}"
        )


    # ======================== Time Window ========================
    for tw in time_windows:
        i, j = tw['train'], tw['op_idx']
        lb, ub = tw['start_lb'], tw['start_ub']
        model.addConstr(t[i, j] >= lb - M * (1 - active[i, j]), name=f"time_lb_{i}_{j}")
        model.addConstr(t[i, j] <= ub + M * (1 - active[i, j]), name=f"time_ub_{i}_{j}")

    # ======================== Successor Constrints ========================
    for op in operations:
        i, j = op['train'], op['op_idx']
        succs = op['successors']
        if succs:
            model.addConstr(gp.quicksum(y[i, j, s] for s in succs) == 1, name=f"succ_choice_{i}_{j}")
        for s in succs:
            # If the successor is selected, the successor operation must be active
            model.addConstr(y[i, j, s] <= active[i, s], name=f"succ_active_link_{i}_{j}_{s}")
            model.addConstr(
                t[i, s] >= t[i, j] + op['min_duration'] +  - M * (1 - y[i, j, s]),
                name=f"succ_time_flow_{i}_{j}_{s}"
            )

    # ======================== Heuristics improvement1: predecessor 约束 ========================
    for op in operations:
        i, j = op['train'], op['op_idx']
        preds = op.get('predecessors', [])  # 你需要在读取阶段预处理出 predecessors
        if preds:
            model.addConstr(active[i, j] <= gp.quicksum(y[p_i, p_j, j] for (p_i, p_j) in preds), 
                            name=f"active_from_preds_{i}_{j}")
        else:
            # 起点直接 active
            model.addConstr(active[i, j] == 1, name=f"source_active_{i}_{j}")

    # ======================== Active  ========================
    for op in operations:
        i, j = op['train'], op['op_idx']
        model.addConstr(t[i, j] >= op['start_lb'] - M * (1 - active[i, j]), name=f"t_lb_active_{i}_{j}")
        model.addConstr(t[i, j] <= op['start_ub'] + M * (1 - active[i, j]), name=f"t_ub_active_{i}_{j}")

##    # ======================== Swapping Conflict ========================
##    for train_a in range(len(trains)):
##        for train_b in range(train_a + 1, len(trains)):
##            # 获取 A、B 的操作列表
##            ops_a = [op for op in operations if op['train'] == train_a]
##            ops_b = [op for op in operations if op['train'] == train_b]
##
##            # 枚举 A 的连续操作 (A1→A2)
##            for idx1 in range(len(ops_a) - 1):
##                a1, a2 = ops_a[idx1], ops_a[idx1 + 1]
##                if len(a1['resources']) == 1 and len(a2['resources']) == 1:
##                    ra1 = a1['resources'][0]
##                    ra2 = a2['resources'][0]
##
##                    # 枚举 B 的连续操作 (B1→B2)
##                    for idx2 in range(len(ops_b) - 1):
##                        b1, b2 = ops_b[idx2], ops_b[idx2 + 1]
##                        if len(b1['resources']) == 1 and len(b2['resources']) == 1:
##                            rb1 = b1['resources'][0]
##                            rb2 = b2['resources'][0]
##
##                            # 判断是否为对向（swapping）资源使用
##                            if ra1 == rb2 and ra2 == rb1 and ra1 != ra2:
##                                # A的最后一步释放 ra2 后，B 才能占用 rb1=ra2
##                                dur = a2['min_duration']
##                                rel = 0
##                                if ra2 in a2['resources']:
##                                    idx = a2['resources'].index(ra2)
##                                    rel = a2['resource_release_times'][idx]
##
##                                i, j = a2['train'], a2['op_idx']
##                                k, l = b1['train'], b1['op_idx']
##                                model.addConstr(
##                                    t[i, j] + dur + rel <= t[k, l] + M * (2 - active[i, j] - active[k, l]),
##                                    name=f"swapping_conflict_{i}_{j}_to_{k}_{l}"
##                                )

    # ======================== Objective function ========================
    obj = gp.LinExpr()
    for obj_item in objectives:
        i = obj_item['train']
        j = obj_item['operation']
        t_bar = obj_item['threshold']
        c_ij = obj_item['coeff']
        d_ij = obj_item['increment']

        delay_var = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"delay_train{i}_op{j}")
        diff_var = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"diff_{i}_{j}")
        model.addConstr(diff_var == t[i, j] - t_bar, name=f"calc_diff_{i}_{j}")
        model.addGenConstrMax(delay_var, [0, diff_var], name=f"gen_max_delay_{i}_{j}")
        obj += c_ij * delay_var

        if d_ij > 0:
            bin_var = model.addVar(vtype=GRB.BINARY, name=f"penalty_trigger_{i}_{j}")
            model.addGenConstrIndicator(bin_var, True, t[i, j] >= t_bar + 1e-5, name=f"penalty_ind_{i}_{j}")
            obj += d_ij * bin_var

    for op in operations:
        i, j = op['train'], op['op_idx']
        obj += 0.0001 * active[i, j]

    model.setObjective(obj, GRB.MINIMIZE)
    return model, t, active, y
