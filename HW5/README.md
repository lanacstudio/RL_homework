# HW4 Deep reinforcement learning in action code (chap3) using

1.  試跑一下 HW_Template=dqn_...OK(已修正).ipynb Code, 讓他正常run 訓練完成
2.  use chatgpt 改寫chap 3.8 第三章 to Pytorch lightning
3. 加上call back (Tensorboard, early stop, dump best model, etc)

**作業要求**： 

    (1) 試跑一下 HW_Template=dqn_...OK(已修正).ipynb Code, 讓他正常run 訓練完成
    
    (2) use chatgpt 改寫chap 3.8 第三章 to Pytorch lightning

    (3) 加上call back (Tensorboard, early stop, dump best model, etc)
      
**✅ (1) 的完整版 code ＆ 跑完的結果都在 🗂️HW45/HW5 DQN PyTorch.ipynb 裡**
**✅ (2)、(3) 的完整版 code ＆ 跑完的結果都在 🗂️HW45/HW5_38_pytorch.ipynb 裡**



## 1. 試跑一下 HW_Template=dqn_...OK(已修正).ipynb Code, 讓他正常run 訓練完成

**(1) ipnyb 檔裡的片段截圖 （完整 code 在 🗂️HW45/HW5 DQN PyTorch.ipynb）**

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW5/imgs/screenshot1.png" width="60%"/>

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW5/imgs/screenshot2.png" width="60%"/>

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW5/imgs/screenshot3.png" width="60%"/>

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW5/imgs/screenshot4.png" width="60%"/>


**(2) 附上部分code with 註解**


```python

# 引入必要的函式庫
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from pytorch_lightning import LightningModule

# 定義 DeepQLearning 類別，繼承自 LightningModule
class DeepQLearning(LightningModule):

  # 初始化方法
  def __init__(self, env_name, policy=epsilon_greedy, capacity=100_000, batch_size=256, lr=1e-3,
               hidden_size=128, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
               eps_start=1.0, eps_end=0.15, eps_last_episode=100, samples_per_epoch=10_000, sync_rate=10):

    super().__init__()

    # 創建環境
    self.env = create_environment(env_name)
    obs_size = self.env.observation_space.shape[0]  # 觀察空間大小
    n_actions = self.env.action_space.n  # 行動空間大小

    # 創建 Q 網路和目標 Q 網路
    self.q_net = DQN(hidden_size, obs_size, n_actions)
    self.target_q_net = copy.deepcopy(self.q_net)

    self.policy = policy  # 使用的策略
    self.buffer = ReplayBuffer(capacity=capacity)  # 經驗回放緩衝區
    self.save_hyperparameters()  # 儲存超參數

    # 在緩衝區中補滿指定數量的樣本
    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f"{len(self.buffer)} 個樣本在經驗緩衝區中。正在補滿...")
      self.play_episode(epsilon=self.hparams.eps_start)

  # 無梯度計算的方法，模擬一個回合的遊玩過程
  @torch.no_grad()
  def play_episode(self, policy=None, epsilon=0.):
    state = self.env.reset()  # 重置環境狀態
    done = False

    while not done:
      if policy:
        action = policy(state, self.env, self.q_net, epsilon=epsilon)  # 根據策略選擇動作
      else:
        action = self.env.action_space.sample()  # 隨機選擇動作

      next_state, reward, done, info = self.env.step(action)  # 執行動作並觀察下一個狀態和獎勵
      exp = (state, action, reward, done, next_state)  # 創建一個經驗元組
      self.buffer.append(exp)  # 將經驗元組加入經驗緩衝區
      state = next_state  # 更新當前狀態為下一個狀態

  # 前向傳播方法
  def forward(self, x):
    return self.q_net(x)

  # 配置優化器
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
    return [q_net_optimizer]

  # 訓練數據加載器
  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
    return dataloader

  # 訓練步驟
  def training_step(self, batch, batch_idx):
    states, actions, rewards, dones, next_states = batch
    actions = actions.unsqueeze(1)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)

    state_action_values = self.q_net(states).gather(1, actions)  # 計算狀態動作值
    next_action_values, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)  # 計算目標 Q 網路的最大動作值

    next_action_values[dones] = 0.0  # 將終止狀態的目標 Q 值設為 0

    expected_state_action_values = rewards + self.hparams.gamma * next_action_values  # 計算預期狀態動作值
    loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)  # 計算損失值
    self.log('episode/Q-Error', loss)  # 記錄損失值到日誌
    return loss

  # 每個訓練周期結束後執行的方法
  def training_epoch_end(self, training_step_outputs):
    epsilon = max(
      self.hparams.eps_end,
      self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode
    )
    self.play_episode(policy=self.policy, epsilon=epsilon)  # 使用更新的策略進行一次遊玩
    self.log('episode/Return', self.env.return_queue[-1])  # 記錄回合的回報值到日誌

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目標 Q 網路的參數

# 定義 ReplayBuffer 類別，用於存儲和取樣經驗
class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.buffer = []

  def __len__(self):
    return len(self.buffer)

  def append(self, experience):
    self.buffer.append(experience)
    if len(self.buffer) > self.capacity:
      self.buffer.pop(0)

  def sample(self, batch_size):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    batch = [self.buffer[idx] for idx in indices]
    return zip(*batch)

# 定義 RLDataset 類別，用於生成訓練數據集
class RLDataset(Dataset):
  def __init__(self, buffer, samples_per_epoch):
    self.buffer = buffer
    self.samples_per_epoch = samples_per_epoch

  def __len__(self):
    return self.samples_per_epoch

  def __getitem__(self, idx):
    return self.buffer[idx]

# 定義 DQN 類別，用於表示深度 Q 網路
class DQN(torch.nn.Module):
  def __init__(self, hidden_size, obs_size, n_actions):
    super(DQN, self).__init__()
    self.fc1 = torch.nn.Linear(obs_size, hidden_size)
    self.fc2 = torch.nn.Linear(hidden_size, n_actions)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# 定義 epsilon_greedy 策略函數
def epsilon_greedy(state, env, q_net, epsilon=0.1):
  if torch.rand(1) < epsilon:
    return env.action_space.sample()
  else:
    with torch.no_grad():
      q_values = q_net(torch.tensor(state, dtype=torch.float32))
      return torch.argmax(q_values).item()

# 創建環境的函數（這裡僅為示例，實際使用時需替換為適合的環境創建方法）
def create_environment(env_name):
  return gym.make(env_name)


```




## 2. use chatgpt 改寫chap 3.8 第三章 to Pytorch lightning
    
**✅ 的完整版 code ＆ 跑完的結果都在 🗂️HW5/HW5_38_pytorch.ipynb 裡 (只留3.8的code)**
**🔆 寫了兩版改寫，分別為 `code(1)` 跟 `code(2)`**


### `code(1)` with 註解**


**(1) 載入 pytorch 等所需套件**


```python
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F

```

**(2) 定義動作＆class**


```python
# 假設動作集合已定義
action_set = ['上', '下', '左', '右']

# 定義 Q 網路架構
class QNet(pl.LightningModule):
    def __init__(self, input_dim=64, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 第一層全連接層
        self.fc2 = nn.Linear(128, 64)         # 第二層全連接層
        self.fc3 = nn.Linear(64, output_dim)  # 輸出層

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 激活函數
        x = F.relu(self.fc2(x))  # ReLU 激活函數
        return self.fc3(x)       # 返回最終輸出

    def training_step(self, batch, batch_idx):
        # 定義訓練步驟，這裡可以根據需要修改
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)  # 記錄訓練損失
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
```


**(3) 初始化 LightningModule 和 Trainer**
 
```python
gridworld_model = QNet()
early_stopping = EarlyStopping(monitor='train_loss', patience=5, mode='min')
gridworld_trainer = pl.Trainer(max_epochs=5000, gpus=1, callbacks=[early_stopping])
```


**(4) 開始訓練**
 
```python
# 虛擬的訓練循環來生成損失
losses = []
for epoch in range(1, 5001):
    # 模擬生成符合模型要求的隨機輸入和標籤
    x = torch.randn(1, 64)  # 假設輸入維度為 (1, 64)
    y = torch.randint(0, 4, (1,))  # 假設標籤維度為 (1,)

    # 使用 trainer 的 training_step 方法進行訓練步驟
    output = gridworld_model.training_step((x, y), epoch)
    
    # 檢查早停條件
    if gridworld_trainer.should_stop:
        print("Early stopping criteria met")
        break

    # 模擬印出準確率等資訊
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/5000], Loss: {output:.4f}")

    # 將損失記錄下來
    losses.append(output.item())
```



**(5) 繪圖**
 
```python
# 繪製訓練損失曲線
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Variation"')
plt.legend()
plt.grid(True)
plt.show()
```


**(6) 截圖**

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW5/imgs/code1.png" width="60%"/>


### `code(2)` with 註解 (only 重要的程式碼區塊，完整版要看 .ipynb)**



**(1) Q 網路模型定義 (QNet class)**

```python
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 第一層全連接層
        self.fc2 = nn.Linear(128, 64)         # 第二層全連接層
        self.fc3 = nn.Linear(64, output_dim)  # 輸出層

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 激活函數
        x = F.relu(self.fc2(x))  # ReLU 激活函數
        return self.fc3(x)       # 返回最終輸出

```

**(2) LightningModule 定義 (LightningGridworld class)**

```python
class LightningGridworld(pl.LightningModule):
    def __init__(self, size=4, mem_size=1000, batch_size=200, sync_freq=500, gamma=0.9, max_moves=50, epsilon_decay=1/5000):
        super(LightningGridworld, self).__init__()
        self.size = size
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.sync_freq = sync_freq
        self.gamma = gamma
        self.max_moves = max_moves
        self.epsilon_decay = epsilon_decay

        # 定義模型
        self.model = QNet(64, 4)
        self.model2 = QNet(64, 4).eval()  # 目標網路

        # 定義優化器和損失函數
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

        # 初始化回放記憶
        self.replay = deque(maxlen=self.mem_size)

        # 其他變數
        self.epsilon = 1.0

    def forward(self, x):
        return self.model(x)

```

**(3) 訓練步驟 (training_step method)**

```python
    def training_step(self, batch, batch_idx):
        state1_batch, action_batch, reward_batch, state2_batch, done_batch = batch

        Q1 = self.model(state1_batch)
        with torch.no_grad():
            Q2 = self.model2(state2_batch)

        Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())

        self.log('train_loss', loss)  # 使用 self.log 記錄訓練損失

        return loss

```

**(4) 驗證步驟 (validation_step method)**

```python
    def validation_step(self, batch, batch_idx):
        state1_batch, action_batch, reward_batch, state2_batch, done_batch = batch

        Q1 = self.model(state1_batch)
        with torch.no_grad():
            Q2 = self.model2(state2_batch)

        Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())

        self.log('val_loss', loss)  # 使用 self.log 記錄驗證損失

```

**(5) 設置早停機制 (EarlyStopping callback)**

```python
# 初始化 LightningGridworld 和 Trainer
gridworld_model = LightningGridworld()
early_stop_callback = pl.callbacks.EarlyStopping(monitor='avg_val_loss', patience=5, verbose=True, mode='min')  # 定義 early stop 機制
gridworld_trainer = pl.Trainer(max_epochs=5000, gpus=1, callbacks=[early_stop_callback])  # 如果有GPU則啟用GPU訓練

```

**(6) 訓練模型 (fit 方法)**

```python
# 訓練模型
gridworld_trainer.fit(gridworld_model)

```


**(7) 截圖**

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW5/imgs/code2.png" width="60%"/>




## (3) 加上call back (Tensorboard, early stop, dump best model, etc)

加入 early stop 的 code 片段：

**`code (1)`:**
 
```python
gridworld_model = QNet()
early_stopping = EarlyStopping(monitor='train_loss', patience=5, mode='min')
gridworld_trainer = pl.Trainer(max_epochs=5000, gpus=1, callbacks=[early_stopping])
```

**`code (2)`:**

```python
gridworld_model = LightningGridworld()
early_stop_callback = pl.callbacks.EarlyStopping(monitor='avg_val_loss', patience=5, verbose=True, mode='min')  # 定義 early stop 機制
gridworld_trainer = pl.Trainer(max_epochs=5000, gpus=1, callbacks=[early_stop_callback])  # 如果有GPU則啟用GPU訓練
```
