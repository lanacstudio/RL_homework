# HW6 Advance DRL in final project you use (找Github or yourown)


# OUTLINE

    🔷 0. 專案 code
    🔷 1. 期末專案簡介
    🔷 2. Q-Learning 及 ε-greedy 介紹 & 本專案使用的 RL code
    🔷 3. 相關 Github 分享

## 🔷 0. 專案 code

### 期末專案 NLPower Assisant：

1. 專案 GitHub 連結： [NLPowerAsistant](https://github.com/lanac0911/NLPowerAsistant)

2. 強化式學習的 code： [NLPowerAsistant/test/anomaly detection.py](https://github.com/lanac0911/NLPowerAsistant/tree/main/test)

## 🔷 1. 期末專案簡介

### 1️⃣. 介紹

   通過結合 NILM 與 NLP 的技術實現一個 user-friendly 的溝通介面，該介面可以實現智慧家居的操作。

  ![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/nlp.png)


 ### 2️⃣. 額外功能

  * 結合了 **強化式學習** 的技術做 **異常偵測（Anomaly Detection）**
  * 加入 「節約提醒」、「異常報告」功能

 ![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/screen.png)


 ### 3️⃣. Flow Chart

  * 結合了 **強化式學習** 的技術做 **異常偵測（Anomaly Detection）**
  * 加入 「節約提醒」、「異常報告」功能

 ![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/flow.png)


 ### 4️⃣. 介面截圖

 ![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/screen2.png)

## 🔷 2. Q-Learning 及 ε-greedy 介紹 & 本專案使用的 RL code

### ⬛️ Q-Learning

### &emsp; 1️⃣. 介紹


Q-learning是一種無模型（model-free）的強化學習算法，用於找出給定環境中的最優動作策略。

是通過學習狀態-行動值函數 
$Q(s,a)$ 來找出最優策略。在每個時間步，智能體觀察當前狀態 $s$，選擇並執行行動 $a$，然後根據獲得的獎勵 $r$ 和到達的新狀態 $s′$ 更新 $Q$ 值。最終，智能體學會在每個狀態下選擇能最大化累積獎勵的行動。

![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/Q-learning.png)

更新公式：

![Q-func](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/q-func.png)

其中：

* 當前 $Q$ 值： $Q(s,a)$
* 學習率 α：決定了新信息在更新過程中的權重，範圍是 0 到 1 之間。高學習率意味著更依賴新信息，低學習率意味著更依賴已有 Q 值。
* 即時獎勵 $r$：智能體在當前狀態 $s$ 執行行動 $a$ 後獲得的獎勵。
* 折扣因子 γ：範圍是 0 到 1 之間，用於平衡即時獎勵和未來獎勵。高折扣因子意味著更重視未來獎勵，低折扣因子則更重視即時獎勵。
* 最大未來 Q 值 $max a' Q(s',a')$ 在下一狀態 $s'$ 下的所有可能行動中選擇最大的 $Q$ 值。

### &emsp; 2️⃣. 專案的 RL code


```python
class Agent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((num_states, num_actions))  # 初始化Q表為零
        self.learning_rate = learning_rate  # 學習率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率

    # Q-learning的部分：更新Q值
    def update_q_table(self, state, action, reward, next_state):
        if next_state < self.q_table.shape[0]:  # 確保索引在邊界之內
            predict = self.q_table[state, action]  # 當前Q值
            target = reward + self.discount_factor * np.max(self.q_table[next_state, :])  # 目標Q值
            self.q_table[state, action] += self.learning_rate * (target - predict)  # 更新Q值

```

predict是當前的 $Q$ 值，target是基於獎勵和下一個狀態的最大 $Q$ 值計算出的目標值。 $Q$ 值通過學習率更新公式進行更新

### ⬛️ ε-greedy 

### &emsp; 1️⃣. 介紹

ε-greedy 策略用於決定智能體在每個狀態下的行動選擇，以平衡探索和利用。探索是指嘗試新的或不常用的行動，以發現更多可能的高獎勵行動。利用是指選擇當前已知的最佳行動，以獲得最大即時獎勵。

![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/greedy.png)


### &emsp; 2️⃣. 專案的 RL code


```python
    # ε-greedy策略的部分：選擇動作
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:  # ε的概率下選擇隨機動作（探索）
            return np.random.choice(self.q_table.shape[1])
        else:  # 1-ε的概率下選擇Q值最大的動作（利用）
            return np.argmax(self.q_table[state, :])
```



### ⬛️ 綜合 Q-Learning 和 ε-greedy 

```python
# 定義智能體類
class Agent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((num_states, num_actions))  # 初始化Q表為零
        self.learning_rate = learning_rate  # 學習率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率

    # ε-greedy策略的部分：選擇動作
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:  # ε的概率下選擇隨機動作（探索）
            return np.random.choice(self.q_table.shape[1])
        else:  # 1-ε的概率下選擇Q值最大的動作（利用）
            return np.argmax(self.q_table[state, :])

    # Q-learning的部分：更新Q值
    def update_q_table(self, state, action, reward, next_state):
        if next_state < self.q_table.shape[0]:  # 確保索引在邊界之內
            predict = self.q_table[state, action]  # 當前Q值
            target = reward + self.discount_factor * np.max(self.q_table[next_state, :])  # 目標Q值
            self.q_table[state, action] += self.learning_rate * (target - predict)  # 更新Q值

```

    
    
## 🔷 3. 相關 Github 分享

1️⃣. [網址](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/git1.png)

2️⃣. [網址](https://github.com/vmayoral/basic_reinforcement_learning)

![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/git2.png)

3️⃣. [網址](https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python)

![Q-learning](https://github.com/lanacstudio/RL_homework/blob/main/HW6(Group)/imgs/git3.png)

