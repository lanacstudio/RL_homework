from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 默认网格大小
grid_size = 5
# 默认起始和结束位置
start_pos = None
end_pos = None
# 默认障碍物位置
obstacles = set()

@app.route('/')
def index():
    return render_template('index.html', grid_size=grid_size)
    # return render_template('HW1-1.html', grid_size=grid_size, start_pos=start_pos, end_pos=end_pos, obstacles=obstacles)

@app.route('/update_grid_size', methods=['POST'])
def update_grid_size():
    global grid_size
    grid_size = int(request.form['gridSize'])
    return jsonify({'success': True})

@app.route('/set_start_end', methods=['POST'])
def set_start_end():
    global start_pos, end_pos
    start_pos = tuple(map(int, request.form['start'].split(',')))
    end_pos = tuple(map(int, request.form['end'].split(',')))
    return jsonify({'success': True})

@app.route('/set_obstacle', methods=['POST'])
def set_obstacle():
    global obstacles
    print('帳案')
    obstacle_pos = tuple(map(int, request.form['obstacle'].split(',')))
    obstacles.add(obstacle_pos)
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
