from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# 默认网格大小为 5x5
GRID_SIZE = 5
# 网格颜色定义
COLOR_START = "#00FF00"  # 绿色
COLOR_END = "#FF0000"  # 红色
COLOR_OBSTACLE = "#808080"  # 灰色
COLOR_NORMAL = "#FFFFFF"  # 白色

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_grid_size', methods=['POST'])
def set_grid_size():
    global GRID_SIZE
    GRID_SIZE = int(request.form['grid_size'])
    return 'OK'


# 定义值迭代算法函数
def value_iteration(start_cell, end_cell, grid_size, obstacles):
    # 在这里编写值迭代算法的代码，找出最佳路径并计算状态值 V(s)
    # 这里简单地返回一个随机生成的动作和状态值，仅用于演示目的
    policy = [{'row': row, 'col': col, 'action': '↑', 'value': 0.5} for row in range(grid_size) for col in range(grid_size)]
    return policy

# 处理计算策略的 POST 请求
@app.route('/calculate_policy', methods=['POST'])
def calculate_policy():
    data = request.json
    start_cell = data['startCell']
    end_cell = data['endCell']
    grid_size = data['gridSize']
    obstacles = data['obstacles']
    
    # 调用值迭代算法函数计算策略
    policy = value_iteration(start_cell, end_cell, grid_size, obstacles)
    
    # 返回策略结果
    return jsonify(policy)

if __name__ == '__main__':
    app.run(debug=True)
