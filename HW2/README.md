# HW2 Q-Learning 路徑規劃 Flask 應用程式

這個 Flask 應用程式可以用於在一個二維網格上進行路徑規劃，並使用 Q-Learning 算法來學習最佳路徑。

### 作業說明

HW2-1:

1. I intend to develop a grid map of size nxn,allow user to  specify the dimension 'n' (from 3 to 7) , using a Flask web application .

2. Users should be able to designate the starting cell (marked as green color after clicking) and the ending cell (marked as red color after click) through mouse clicks
3. setting n-2 obstales by clicking mouse and turn those clicked cell as grey.

HW2-2: 

1. show the random generated action (with up, down, left, and right arrows) for each cell as policy

2. use policy evalation to derive value V(s) for each state

 

HW2-3 

1. use value Q-Learning to derive the optimal policy and display actions 
2. show V(s) on each cell


## 程式碼

1. 首先確保已安裝 Flask 和 numpy 套件：

    ```
    pip install Flask numpy
    ```

2. 運行 Flask 應用程式：

    ```
    python app.py
    ```

3. 打開瀏覽器訪問 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

4. 在網格頁面上設定起始點、終止點和障礙物，然後點擊"開始搜索"按鈕。

5. 程式將根據 Q-Learning 算法計算出最佳路徑，並在網格上顯示出來。

## 檔案結構

- `app.py`: Flask 應用程式的主要檔案，包含了網頁路由和 Q-Learning 算法實現。
- `templates/HW2.html`: 網頁模板檔案，定義了網格頁面的佈局和交互。
- `static/style.css`: 網頁的樣式表檔案，定義了網格頁面的樣式。

## 注意事項

- 如果需要修改網格大小、起始點、終止點或障礙物，請在網頁上進行設定，並點擊"開始搜索"按鈕。
- 程式會自動使用 Q-Learning 算法計算最佳路徑，無需手動觸發。
- 網格大小預設為 5x5，可以通過修改程式碼中的 `grid_size` 變數來更改。

## HW2

### Flask Web 截圖

<img src="https://github.com/lanacstudio/RL_homework/blob/main/result/r3.png" width="80%"/>
<img src="https://github.com/lanacstudio/RL_homework/blob/main/result/r1.png" width="80%"/>
<img src="https://github.com/lanacstudio/RL_homework/blob/main/result/r2.png" width="80%"/>


### Demo 影片
[Demo 影片](https://github.com/lanacstudio/RL_homework/blob/main/result/demo_video.mov)

### Code

在 ```app.py``` 跟 ```HW1-3.html```

``` python
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 默認網格大小
grid_size = 5
# 默認起始和結束位置
start_pos = None
end_pos = None
# 默認障礙物位置
obstacles = set()

# 定義狀態值函數和策略
Q = np.zeros((grid_size, grid_size, 4))  # 狀態動作值函數
policy = np.zeros((grid_size, grid_size), dtype=int)  # 最佳策略，0: 上, 1: 下, 2: 左, 3: 右

# 定義 Q-Learning 算法參數
alpha = 0.1  # 學習率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率

# 初始化 Q-Learning 算法
def init_q_learning():
    global Q
    Q = np.zeros((grid_size, grid_size, 4))  # 重置狀態動作值函數

# 更新 Q-Learning 算法
def update_q_learning(current_pos, next_pos, action, reward):
    max_next_q = np.max(Q[next_pos])
    Q[current_pos][action] += alpha * (reward + gamma * max_next_q - Q[current_pos][action])

# 根據 Q-Learning 算法選擇動作
def choose_action(state):
    if np.random.uniform() < epsilon:
        return np.random.randint(0, 4)  # 隨機選擇動作
    else:
        return np.argmax(Q[state])

# 更新策略
def update_policy():
    global policy
    for i in range(grid_size):
        for j in range(grid_size):
            policy[i, j] = np.argmax(Q[i, j])

# 更新網格大小
@app.route('/update_grid_size', methods=['POST'])
def update_grid_size():
    global grid_size
    grid_size = int(request.form['gridSize'])
    return jsonify({'success': True})

# 設置起始和結束位置
@app.route('/set_start_end', methods=['POST'])
def set_start_end():
    global start_pos, end_pos
    start_pos = tuple(map(int, request.form['start'].split(',')))
    end_pos = tuple(map(int, request.form['end'].split(',')))
    init_q_learning()  # 重置 Q-Learning 算法
    return jsonify({'success': True})

# 設置障礙物
@app.route('/set_obstacle', methods=['POST'])
def set_obstacle():
    global obstacles
    obstacle_pos = tuple(map(int, request.form['obstacle'].split(',')))
    obstacles.add(obstacle_pos)
    return jsonify({'success': True})

# 獲取網格信息
@app.route('/get_grid_info', methods=['GET'])
def get_grid_info():
    return jsonify({
        'grid_size': grid_size,
        'start_pos': start_pos,
        'end_pos': end_pos,
        'obstacles': list(obstacles)
    })

# 執行 Q-Learning 算法
def q_learning_algorithm():
    global Q
    for episode in range(1000):
        current_pos = start_pos
        while current_pos != end_pos:
            action = choose_action(current_pos)
            next_pos = get_next_position(current_pos, action)
            reward = get_reward(next_pos)
            update_q_learning(current_pos, next_pos, action, reward)
            current_pos = next_pos
    update_policy()

# 獲取下一個位置
def get_next_position(current_pos, action):
    i, j = current_pos
    if action == 0:  # 上
        return (max(i - 1, 0), j)
    elif action == 1:  # 下
        return (min(i + 1, grid_size - 1), j)
    elif action == 2:  # 左
        return (i, max(j - 1, 0))
    elif action == 3:  # 右
        return (i, min(j + 1, grid_size - 1))

# 獲取獎勵
def get_reward(next_pos):
    if next_pos == end_pos:
        return 1  # 到達目標位置，給予正獎勵
    elif next_pos in obstacles:
        return -1  # 碰到障礙物，給予負獎勵
    else:
        return 0

# 執行 Q-Learning 算法
q_learning_algorithm()

# 主頁
@app.route('/')
def index():
    return render_template('./HW2.html', grid_size=grid_size, start_pos=start_pos, end_pos=end_pos, obstacles=obstacles)

if __name__ == '__main__':
    app.run(debug=True)

```
