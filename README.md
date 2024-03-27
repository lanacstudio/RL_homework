## HW1

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

### Flask Web 截圖

<img src="https://github.com/lanacstudio/RL_homework/blob/main/result/r3.png" width="80%"/>
<img src="https://github.com/lanacstudio/RL_homework/blob/main/result/r1.png" width="80%"/>
<img src="https://github.com/lanacstudio/RL_homework/blob/main/result/r2.png" width="80%"/>


### Demo 影片
[Demo 影片](https://github.com/lanacstudio/RL_homework/blob/main/result/demo_video.mov)

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
