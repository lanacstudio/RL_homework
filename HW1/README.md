# HW1 Iteration Algorith 路徑規劃 Flask 應用程式

這個 Flask 應用程式可以用於在一個二維網格上進行路徑規劃，並使用 Iteration Algorith 算法來學習最佳路徑。

### 作業說明

HW1-1:

1. I intend to develop a grid map of size nxn,allow user to  specify the dimension 'n' (from 3 to 7) , using a Flask web application .

2. Users should be able to designate the starting cell (marked as green color after clicking) and the ending cell (marked as red color after click) through mouse clicks
3. setting n-2 obstales by clicking mouse and turn those clicked cell as grey.

HW1-2: 

1. show the random generated action (with up, down, left, and right arrows) for each cell as policy

2. use policy evalation to derive value V(s) for each state

 

HW1-3 

1. use value iteration algorithm to derive the optimal policy and display actions 
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

### Flask Web 截圖

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW1/result/r3.png" width="60%"/>
<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW1/result/r1.png" width="60%"/>
<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW1/result/r1.png" width="60%"/>

### Demo 影片
[Demo 影片](https://github.com/lanacstudio/RL_homework/blob/main/HW1/result/demo_video.mov)

### Code

在 ```app.py``` 跟 ```HW1-3.html```

``` python
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
```
