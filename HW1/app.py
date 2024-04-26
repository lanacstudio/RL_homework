from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 預設網格大小
grid_size = 5
# 預設起始和結束位置
start_pos = None
end_pos = None
# 預設障礙物位置
obstacles = set()

# 定義狀態值函數和策略
V = np.zeros((grid_size, grid_size))  # 狀態值函數
policy = np.zeros((grid_size, grid_size), dtype=int)  # 最佳策略，0: 上, 1: 下, 2: 左, 3: 右

# 定義值迭代函數
def value_iteration():
    global V, policy
    # 值迭代的參數
    gamma = 0.9  # 折扣因子
    epsilon = 1e-6  # 收斂閾值

    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == start_pos or (i, j) == end_pos:
                    continue  # 起始點和結束點的值不需要更新
                v = V[i, j]
                # 計算當前狀態的最大價值動作
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
                    # 檢查下一個狀態是否有效
                    if next_i >= 0 and next_i < grid_size and next_j >= 0 and next_j < grid_size and (next_i, next_j) not in obstacles:
                        value = 0 if (next_i, next_j) == end_pos else gamma * V[next_i, next_j]
                        max_value = max(max_value, value)
                # 更新狀態值函數和策略
                V[i, j] = max_value
                delta = max(delta, abs(v - V[i, j]))
        # 如果狀態值函數收斂，則退出迴圈
        if delta < epsilon:
            break

    # 根據狀態值函數確定最佳策略
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == start_pos or (i, j) == end_pos:
                continue  # 起始點和結束點無需確定策略
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
                # 檢查下一個狀態是否有效
                if next_i >= 0 and next_i < grid_size and next_j >= 0 and next_j < grid_size and (next_i, next_j) not in obstacles:
                    value = 0 if (next_i, next_j) == end_pos else gamma * V[next_i, next_j]
                    if value > max_value:
                        max_value = value
                        max_action = action
            # 更新最佳策略
            policy[i, j] = max_action

@app.route('/')
def index():
    return render_template('HW1-3.html', grid_size=grid_size, start_pos=start_pos, end_pos=end_pos, obstacles=obstacles)

@app.route('/update_grid_size', methods=['POST'])
def update_grid_size():
    global grid_size
    grid_size = int(request.form['gridSize'])
    return jsonify({'success': True})

# 繼續修改 Flask 應用
@app.route('/set_start_end', methods=['POST'])
def set_start_end():
    global start_pos, end_pos
    start_pos = tuple(map(int, request.form['start'].split(',')))
    end_pos = tuple(map(int, request.form['end'].split(',')))
    # 執行值迭代演算法
    value_iteration()
    return jsonify({'success': True})

@app.route('/set_obstacle', methods=['POST'])
def set_obstacle():
    global obstacles
    obstacle_pos = tuple(map(int, request.form['obstacle'].split(',')))
    obstacles.add(obstacle_pos)
    # 執行值迭代演算法
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
