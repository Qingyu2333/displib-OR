import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ====== 修改为你的 solution 路径 ======
solution_path = r"C:\Users\陆柯言\Desktop\大四第二学期学习资料\应用运筹project\displib_instances_phase1_v1_1\solution\line3_1.json"

# 读取 JSON 文件
with open(solution_path, encoding='utf-8') as f:
    data = json.load(f)

events = data["events"]
objective_value = data.get("objective_value", None)

# 提取列车 ID 并编号
train_ids = sorted(set(event["train"] for event in events))
train_to_y = {train_id: idx for idx, train_id in enumerate(train_ids)}
num_trains = len(train_ids)

# 为每辆列车分配颜色（colormap）
cmap = cm.get_cmap('tab20', num_trains)
train_colors = {train: cmap(i) for i, train in enumerate(train_ids)}

# 设置画布
fig, ax = plt.subplots(figsize=(12, 0.6 * num_trains + 2))

# 绘制每个 operation
for event in events:
    train = event["train"]
    op = event["operation"]
    start_time = event["time"]
    duration = 10  # 可改为真实 duration
    y = train_to_y[train]

    # 条形图
    ax.barh(y, duration, left=start_time, height=0.4,
            color=train_colors[train])



# 坐标轴与标题
ax.set_yticks(list(train_to_y.values()))
ax.set_yticklabels([f"Train {tid}" for tid in train_ids], fontsize=9)
ax.set_xlabel("Time", fontsize=11)
ax.set_title(f"Train Operation Schedule for line3_1(Objective: {objective_value})", fontsize=13, pad=15)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# 图例（只显示前6个）
patches = [plt.Line2D([0], [0], color=train_colors[train], lw=6, label=f'Train {train}') 
           for train in train_ids[:6]]
ax.legend(handles=patches, loc='lower right', title='Trains', fontsize=8)

plt.tight_layout()
plt.show()

# 保存为高清图片（可选）
# plt.savefig("train_schedule_gantt.png", dpi=300)

