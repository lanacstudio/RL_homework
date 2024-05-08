# HW4 Deep reinforcement learning in action code (chap3) using

**作業要求**： 

    (1) 跑一遍作註解 (excliadraw 畫一張圖 DQN update 示意圖 With  circle 1,2,3,4, 並在 Code 內對應)
    
    (2) 跑最難的 random mode 註解即可
        
        (a) 流程圖 + code 解釋對照 for random mode
        
          備註：
              (1) static gridworld (+, -, w, P) 都不變
              (2) player mode P random 
              (3) random mode +,-, W, P 都會改變 
          
**✅（完整版）跑完的結果都在 HW4/HW4 code.ipynb 裡**



## 1. 跑一遍作註解 (excliadraw 畫一張圖 DQN update 示意圖 With  circle 1,2,3,4, 並在 Code 內對應)

**(0) chap3 的 code 跑一遍**

**（完整版）跑完的結果都在 HW4/HW4 code.ipynb 裡**


**(1) DQN update 示意圖**

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW4/DQN-full.png" width="60%"/>


**(2) code with 註解＆對應**


```python
# 導入必要的庫
from collections import deque
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 設定訓練參數
epochs = 5000  # 訓練次數
losses = []  # 儲存每次訓練的損失值
mem_size = 1000  # 記憶體大小
batch_size = 200  # 小批次大小
replay = deque(maxlen=mem_size)  # 建立一個記憶體deque
max_moves = 50  # 最大步數

for i in range(epochs):
    # 初始化遊戲環境
    game = Gridworld(size=4, mode='random')
    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0  # 記錄步數
    while(status == 1): 
        mov += 1
        # 選擇行動
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)     
        action = action_set[action_]
        # 執行行動
        game.makeMove(action)
        # 獲得新狀態和獎勵
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()
        done = True if reward != -1 else False
        # 儲存經驗
        exp = (state1, action_, reward, state2, done)
        replay.append(exp)
        state1 = state2
        # 開始小批次訓練
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch]) 
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])            
            Q1 = model(state1_batch)
            with torch.no_grad():
                Q2 = model(state2_batch)  
            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze() 
            # 計算損失
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 結束條件
        if abs(reward) == 10 or mov > max_moves:
            status = 0
            mov = 0
    # 更新 ε-greedy 的 ε 值
    if epsilon > 0.1: 
        epsilon -= (1/epochs)

# 將損失值轉換為numpy數組
losses = np.array(losses)
# 繪製損失曲線
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Steps",fontsize=11)
plt.ylabel("Loss",fontsize=11)

```
<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW4/DQN-result.png" width="60%"/>




## (2) 跑最難的 random mode 註解即可 (summary + 流程圖 + code 解釋對照)
    
補充：

    (a) static gridworld (+, -, w, P) 都不變
    (b) player mode P random 
    (c) random mode +,-, W, P 都會改變 


**（完整版）跑完的結果都在 HW4/HW4 code.ipynb 裡**

**流程圖:**

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW4/DQN-random mode.png" width="65%"/>


**code:**

```python
# 導入必要的庫
from collections import deque
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 設定訓練參數
epochs = 1000  # 訓練次數
losses = []  # 儲存每次的損失值
epsilon = 1.0  # 初始 ε 值
gamma = 0.99  # 折扣因子
batch_size = 200  # 小批次大小

# 遍歷每個訓練時期
for i in range(epochs):
    # 初始化遊戲環境
    game = Gridworld(size=4, mode='random')
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state1 = torch.from_numpy(state_).float()
    status = 1  # 紀錄遊戲是否繼續
    while(status == 1):
        # 選擇動作
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon): 
            action_ = np.random.randint(0, 4)  # ε-貪婪策略：以 ε 的機率隨機選擇動作
        else:
            action_ = np.argmax(qval_)  # 根據 Q 值選擇動作
        action = action_set[action_]  # 將動作數字轉換為字母
        game.makeMove(action)  # 執行動作
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state2 = torch.from_numpy(state2_).float()  # 獲取新狀態
        reward = game.reward()
        with torch.no_grad(): 
            newQ = model(state2.reshape(1, 64))
        maxQ = torch.max(newQ)  # 獲取新狀態下的最大 Q 值
        if reward == -1:
            Y = reward + (gamma * maxQ)  # 計算目標 Q 值
        else: 
            Y = reward  # 遊戲結束，目標 Q 值即為回饋值
        Y = torch.Tensor([Y]).detach() 
        X = qval.squeeze()[action_]  # 獲取實際動作的預測 Q 值
        loss = loss_fn(X, Y)  # 計算損失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state1 = state2
        if abs(reward) == 10:       
            status = 0  # 若 reward 的絕對值為 10，遊戲結束
    losses.append(loss.item())
    if epsilon > 0.1: 
        epsilon -= (1/epochs)  # 隨著訓練進行，逐漸減小 ε 的值
    # 每 100 次訓練，打印一次損失值
    if i % 100 == 0:
        print(i, loss.item())
        clear_output(wait=True)

# 將損失值繪製成圖
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=11)
plt.ylabel("Loss",fontsize=11)

```

**result:**

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW4/random-result.png" width="60%"/>



**summary:**

目的：訓練一個 DQN 模型，使其能夠從隨機模式下的遊戲環境中學習到最優的動作策略。


  * 初始化設定：設定訓練的參數，如訓練次數（epochs）、記憶體大小（mem_size）、小批次大小（batch_size）等。
  * 訓練迴圈：使用 for 迴圈執行指定次數的訓練。
  * 在每個訓練迴圈中：
      * 初始化遊戲環境，獲取初始狀態。
      * 進入遊戲迴圈，直到遊戲結束：
      * 根據當前狀態，使用 ε-貪婪策略選擇動作：以一定的機率隨機選擇動作，或根據 Q 值選擇最佳動作。
      * 執行選擇的動作，獲得新的狀態和回饋。
      * 將經驗（當前狀態、動作、回饋、新狀態）儲存到記憶體中。
      * 從記憶體中隨機抽取一個小批次的經驗，進行深度 Q 網絡的訓練。
      * 在訓練迴圈的每個時期末將損失值儲存在列表中。
  * 調整 ε 的值：在每個訓練時期末根據訓練進行情況，逐漸降低 ε 的值，以減少探索的機率。
  * 繪製損失曲線：將所有時期的損失值繪製成曲線，以便分析訓練過程的效果。


  
