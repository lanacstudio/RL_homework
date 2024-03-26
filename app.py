from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 默认网格大小
grid_size = 5
# 默认起始和结束位置
start_pos = None
end_pos = None
# 默认障碍物位置
obstacles = set()

# 定义状态值函数和策略
V = np.zeros((grid_size, grid_size))  # 状态值函数
policy = np.zeros((grid_size, grid_size), dtype=int)  # 最优策略，0: 上, 1: 下, 2: 左, 3: 右

# 定义值迭代函数
def value_iteration():
    global V, policy
    # 值迭代的参数
    gamma = 0.9  # 折扣因子
    epsilon = 1e-6  # 收敛阈值

    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == start_pos or (i, j) == end_pos:
                    continue  # 起始点和结束点的值不需要更新
                v = V[i, j]
                # 计算当前状态的最大价值动作
                max_value = float("-inf")
                for action in range(4):  # 上, 下, 左, 右
                    if action == 0:  # 上
                        next_i, next_j = i - 1, j
                    elif action == 1:  # 下
                        next_i, next_j = i + 1, j
                    elif action == 2:  # 左
                        next_i, next_j = i, j - 1
                    elif action == 3:  # 右
                        next_i, next_j = i, j + 1
                    # 检查下一个状态是否有效
                    if next_i >= 0 and next_i < grid_size and next_j >= 0 and next_j < grid_size and (next_i, next_j) not in obstacles:
                        value = 0 if (next_i, next_j) == end_pos else gamma * V[next_i, next_j]
                        max_value = max(max_value, value)
                # 更新状态值函数和策略
                V[i, j] = max_value
                delta = max(delta, abs(v - V[i, j]))
        # 如果状态值函数收敛，则退出循环
        if delta < epsilon:
            break

    # 根据状态值函数确定最优策略
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == start_pos or (i, j) == end_pos:
                continue  # 起始点和结束点无需确定策略
            max_action = None
            max_value = float("-inf")
            for action in range(4):  # 上, 下, 左, 右
                if action == 0:  # 上
                    next_i, next_j = i - 1, j
                elif action == 1:  # 下
                    next_i, next_j = i + 1, j
                elif action == 2:  # 左
                    next_i, next_j = i, j - 1
                elif action == 3:  # 右
                    next_i, next_j = i, j + 1
                # 检查下一个状态是否有效
                if next_i >= 0 and next_i < grid_size and next_j >= 0 and next_j < grid_size and (next_i, next_j) not in obstacles:
                    value = 0 if (next_i, next_j) == end_pos else gamma * V[next_i, next_j]
                    if value > max_value:
                        max_value = value
                        max_action = action
            # 更新最优策略
            policy[i, j] = max_action

@app.route('/')
def index():
    return render_template('HW1-3.html', grid_size=grid_size, start_pos=start_pos, end_pos=end_pos, obstacles=obstacles)

@app.route('/update_grid_size', methods=['POST'])
def update_grid_size():
    global grid_size
    grid_size = int(request.form['gridSize'])
    return jsonify({'success': True})

# 继续修改 Flask 应用
@app.route('/set_start_end', methods=['POST'])
def set_start_end():
    global start_pos, end_pos
    start_pos = tuple(map(int, request.form['start'].split(',')))
    end_pos = tuple(map(int, request.form['end'].split(',')))
    # 运行值迭代算法
    value_iteration()
    return jsonify({'success': True})

@app.route('/set_obstacle', methods=['POST'])
def set_obstacle():
    global obstacles
    obstacle_pos = tuple(map(int, request.form['obstacle'].split(',')))
    obstacles.add(obstacle_pos)
    # 运行值迭代算法
    value_iteration()
    return jsonify({'success': True})

@app.route('/get_grid_info', methods=['GET'])
def get_grid_info():
    return jsonify({
        'grid_size': grid_size,
        'start_pos': start_pos,
        'end_pos': end_pos,
        'obstacles': list(obstacles)
    })

if __name__ == '__main__':
    app.run(debug=True)
