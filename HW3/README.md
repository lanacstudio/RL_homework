# HW3 explore and explit for arm-bandit problem

**作業要求**： 在多臂老虎機問題上探索和利用不同算法的性能。
分別對以下4個演算法

    (1) epsilon-greedy
    (2) UCB
    (3) Softmax
    (4) Thompson Sampling 

進行3種分析：

    (1) 供LaTeX中的算法
    (2) ChatGPT Prompt
    (3) 給出一個用於多臂老虎機問題的Python實現，並附上一個繪製性能的圖表
    (4) 解釋結果，包括在空間和時間上的性能


## 演算法1. Epsilon-Greedy

**(1) algorithm in latex**

```latex
\documentclass{article}
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{document}
\begin{algorithm}
\caption{Epsilon-Greedy算法}
\begin{algorithmic}[1]
\STATE 初始化所有動作$a$的$Q(a)$和$N(a)$
\STATE 設置探索率$\epsilon$
\FOR{$t = 1, 2, ..., T$}
    \IF{以概率$\epsilon$}
        \STATE 隨機選擇動作$a_t$
    \ELSE
        \STATE 選擇$a_t = \arg\max_a Q(a)$
    \ENDIF
    \STATE 執行動作$a_t$，觀察獎勵$r_t$
    \STATE 更新$Q(a_t)$和$N(a_t)$
\ENDFOR
\end{algorithmic}
\end{algorithm}
\end{document}

```


**(2) ChatGPT Prompt**

*Write Python code that implements the Epsilon-Greedy algorithm for a Multi-Armed Bandit (MAB) problem with a specified number of arms, number of trials, and epsilon value. The code should keep track of estimated average rewards for each arm and the number of times each arm has been pulled. Calculate and return the average reward per trial. Additionally, plot the average reward per trial across all trials to visualize the exploration-exploitation trade-off.*

**(3) Code & Plot**

```python
import random
import matplotlib.pyplot as plt

def epsilon_greedy(k, T, epsilon):
  """
  Epsilon-Greedy algorithm for MAB.

  Args:
      k: Number of arms.
      T: Number of trials.
      epsilon: Exploration rate.

  Returns:
      Average reward per trial.
  """
  Q = [0] * k  # Estimated average rewards for each arm
  N = [0] * k  # Times each arm has been pulled
  average_rewards = []

  for t in range(T):
    # Exploration
    if random.random() < epsilon:
      arm_i = random.randint(0, k-1)
    # Exploitation
    else:
      arm_i = max(range(k), key=Q.__getitem__)

    reward = random.random()  # Simulate random reward from arm
    N[arm_i] += 1
    Q[arm_i] = (N[arm_i] * Q[arm_i] + reward) / N[arm_i]

    average_rewards.append(sum(Q) / k)

  plt.plot(average_rewards)
  plt.xlabel("Trials")
  plt.ylabel("Average Reward")
  plt.title("Epsilon-Greedy Algorithm (Exploration-Exploitation Trade-off)")
  plt.show()

  return sum(Q) / k  # Average reward per trial

# Example usage
k = 10  # Number of arms
T = 1000  # Number of trials
epsilon = 0.1  # Exploration rate

average_reward = epsilon_greedy(k, T, epsilon)
print(f"Average reward per trial: {average_reward:.4f}")

```

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW3/result/Epsilon-Greedy.png" width="90%"/>

**(4) 結果解釋：**

**空間上：**

  * 初始階段，epsilon-greedy可能會探索所有手臂以收集基本信息（空間探索）。
    
  * 隨著時間的推移，epsilon-greedy可能會偏好那些一直產生較高獎勵的手臂，將其拉力集中在這些手臂上（空間利用）。

**時間上：**

 * 試驗開始時，探索更加普遍，因為算法試圖了解各個手臂的獎勵分佈。
 * 隨著試驗的進行，利用變得更加突出，優先考慮那些估計平均獎勵最高的手臂。
 * 平均獎勵每試驗的圖表反映了這種時間模式，通常由於探索而呈現出初始上升，隨後由於利用而逐漸穩定或增加。




## 演算法2. UCB (Upper Confidence Bound) 


**(1) algorithm in latex**

```latex
\documentclass{article}
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{document}
\begin{algorithm}
\caption{UCB算法}
\begin{algorithmic}[1]
\STATE 初始化所有動作$a$的$Q(a)$和$N(a)$
\FOR{$t = 1, 2, ..., T$}
    \STATE 選擇$a_t = \arg\max_a \left( Q(a) + c \sqrt{\frac{\ln(t)}{N(a)}} \right)$
    \STATE 執行動作$a_t$，觀察獎勵$r_t$
    \STATE 更新$Q(a_t)$和$N(a_t)$
\ENDFOR
\end{algorithmic}
\end{algorithm}
\end{document}

```


**(2) ChatGPT Prompt**

*Write Python code that implements the UCB (Upper Confidence Bound) algorithm for a Multi-Armed Bandit (MAB) problem with a specified number of arms and number of trials. The code should keep track of estimated average rewards for each arm, the number of times each arm has been pulled, and the UCB values for each arm. Calculate and return the average reward per trial. Additionally, plot the average reward per trial across all trials to visualize the exploration-exploitation trade-off.*

**(3) Code & Plot**

```python
import random
import math
import matplotlib.pyplot as plt

def ucb(k, T):
  """
  UCB algorithm for MAB.

  Args:
      k: Number of arms.
      T: Number of trials.

  Returns:
      Average reward per trial.
  """
  Q = [0] * k  # Estimated average rewards for each arm
  N = [1] * k  # Initialize N[i] to 1 to avoid division by zero (optional)
  average_rewards = []

  for t in range(T):
    # Choose arm with highest UCB value, add offset to log argument (optional)
    UCB_values = [Q[i] + math.sqrt(2 * math.log(max(t, 2)) / N[i]) for i in range(k)]
    arm_i = max(range(k), key=UCB_values.__getitem__)

    reward = random.random()  # Simulate random reward from arm
    N[arm_i] += 1
    Q[arm_i] = (N[arm_i] * Q[arm_i] + reward) / N[arm_i]

    average_rewards.append(sum(Q) / k)

  plt.plot(average_rewards)
  plt.xlabel("Trials")
  plt.ylabel("Average Reward")
  plt.title("UCB Algorithm (Balancing Exploration and Exploitation)")
  plt.show()

  return sum(Q) / k  # Average reward per trial

# Example usage
k = 10  # Number of arms
T = 1000  # Number of trials

average_reward = ucb(k, T)
print(f"Average reward per trial: {average_reward:.4f}")

```

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW3/result/UCB.png" width="90%"/>

**(4) 結果解釋：**

**空間上：**

  * UCB專注於探索具有高不確定性（高UCB值）的臂以及利用具有高估計平均獎勵的臂。
  * 這導致相對於epsilon-greedy，在早期階段更平衡的探索-利用策略。
  * 隨著試驗的進行，如果估計平均獎勵也是有希望的，UCB可能仍然會偏向於具有較高不確定性的臂。
    
**時間上：**

 * UCB的探索-利用平衡隨時間演變。
 * 最初，探索主導，因為算法試圖了解所有臂的獎勵分佈。
 * 隨著收集到更多的數據，利用變得更加重要，但UCB仍然保持一定程度的探索，以可能發現更好的臂。
 * 每次試驗的平均獎勵圖通常顯示出由於探索而導致的初始上升，然後隨著利用的開展而逐漸穩定或增加。







## 演算法3. Softmax


**(1) algorithm in latex**

```latex
\documentclass{article}
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{document}
\begin{algorithm}
\caption{Softmax算法}
\begin{algorithmic}[1]
\STATE 初始化所有動作$a$的$Q(a)$和$N(a)$
\FOR{$t = 1, 2, ..., T$}
    \STATE 計算動作$a$的機率：$P(a) = \frac{e^{Q(a) / \tau}}{\sum_b e^{Q(b) / \tau}}$
    \STATE 根據機率$P(a)$選擇動作$a_t$
    \STATE 執行動作$a_t$，觀察獎勵$r_t$
    \STATE 更新$Q(a_t)$和$N(a_t)$
\ENDFOR
\end{algorithmic}
\end{algorithm}
\end{document}
```


**(2) ChatGPT Prompt**

*Write Python code that implements the Softmax algorithm for a Multi-Armed Bandit (MAB) problem with a specified number of arms, number of trials, and temperature parameter. The code should keep track of estimated average rewards for each arm, the number of times each arm has been pulled, and the probabilities for each arm at each trial. Calculate and return the average reward per trial. Additionally, plot the average reward per trial across all trials to visualize the exploration-exploitation trade-off.*

**(3) Code & Plot**

```python
import random
import math  # Import math for exp function
import matplotlib.pyplot as plt

def softmax(k, T, tau):
  """
  Softmax algorithm for MAB.

  Args:
      k: Number of arms.
      T: Number of trials.
      tau: Temperature parameter (controls exploration-exploitation trade-off).

  Returns:
      Average reward per trial.
  """
  Q = [0] * k  # Estimated average rewards for each arm
  N = [0] * k  # Times each arm has been pulled
  average_rewards = []

  for t in range(T):
    # Calculate probabilities using Softmax function
    probabilities = [math.exp(tau * Q[i] / sum(math.exp(tau * Q[j]) for j in range(k))) for i in range(k)]
    arm_i = random.choices(range(k), weights=probabilities)[0]

    reward = random.random()  # Simulate random reward from arm
    N[arm_i] += 1
    Q[arm_i] = (N[arm_i] * Q[arm_i] + reward) / N[arm_i]

    average_rewards.append(sum(Q) / k)

  plt.plot(average_rewards)
  plt.xlabel("Trials")
  plt.ylabel("Average Reward")
  plt.title(f"Softmax Algorithm (Exploration-Exploitation Trade-off, tau={tau})")
  plt.show()

  return sum(Q) / k  # Average reward per trial

# Example usage
k = 10  # Number of arms
T = 1000  # Number of trials
tau = 1  # Temperature parameter

average_reward = softmax(k, T, tau)
print(f"Average reward per trial: {average_reward:.4f}")

```

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW3/result/Softmax.png" width="90%"/>

**(4) 結果解釋：**

**空間上：**

 * Softmax根據手臂的估計平均獎勵分配較高的概率給估計平均獎勵較高的手臂，給估計平均獎勵較低的手臂分配較低的概率。
 * 溫度參數（tau）控制這些概率的銳度。較高的tau值使概率更加集中，有利於利用。較低的tau值使概率更加分散，允許更多的探索。

    
**時間上：**

 * Softmax中的探索和利用平衡取決於溫度參數（tau）和隨時間變化的估計平均獎勵。
 * 最初，特別是對於較低的tau值，探索可能更普遍。
 * 隨著試驗的進行和估計平均獎勵變得更加準確，利用可能會變得更加突出。




## 演算法4. Thompson Sampling


**(1) algorithm in latex**

```latex
\documentclass{article}
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{document}
\begin{algorithm}
\caption{Thompson Sampling算法}
\begin{algorithmic}[1]
\STATE 初始化所有動作$a$的計數: $N_1(a) = N_2(a) = 0$
\FOR{$t = 1, 2, ..., T$}
    \STATE 根據每個動作$a$的Beta分佈抽樣: $X(a) \sim \text{Beta}(N_1(a) + 1, N_2(a) + 1)$
    \STATE 選擇動作$a_t = \arg\max_a X(a)$
    \STATE 執行動作$a_t$，觀察獎勵$r_t$
    \IF{$r_t = 1$}
        \STATE $N_1(a_t) \leftarrow N_1(a_t) + 1$
    \ELSE
        \STATE $N_2(a_t) \leftarrow N_2(a_t) + 1$
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
\end{document}

```


**(2) ChatGPT Prompt**

*Write Python code that implements the Thompson Sampling algorithm for a Multi-Armed Bandit (MAB) problem with a specified number of arms and number of trials. The code should keep track of estimated mean and standard deviation for each arm, the number of times each arm has been pulled, and the sampled rewards for each arm at each trial. Calculate and return the average reward per trial. Additionally, plot the average reward per trial across all trials to visualize the exploration-exploitation trade-off.*

**(3) Code & Plot**

```python
import random
import numpy as np
import matplotlib.pyplot as plt

def thompson_sampling(k, T):
  """
  Thompson Sampling algorithm for MAB.

  Args:
      k: Number of arms.
      T: Number of trials.

  Returns:
      Average reward per trial.
  """
  mu = [0] * k  # Initial mean estimates for each arm
  sigma = [1] * k  # Initial standard deviation estimates for each arm
  N = [0] * k  # Times each arm has been pulled
  average_rewards = []

  for t in range(T):
    # Draw samples from posterior distributions for each arm
    sampled_rewards = [np.random.normal(mu[i], sigma[i]**2) for i in range(k)]
    arm_i = np.argmax(sampled_rewards)

    reward = random.random()  # Simulate random reward from arm
    N[arm_i] += 1
    mu[arm_i] = (N[arm_i] * mu[arm_i] + reward) / N[arm_i]
    sigma[arm_i] = np.sqrt((N[arm_i] * sigma[arm_i]**2 + (reward - mu[arm_i])**2) / N[arm_i])

    average_rewards.append(sum(mu) / k)

  plt.plot(average_rewards)
  plt.xlabel("Trials")
  plt.ylabel("Average Reward")
  plt.title("Thompson Sampling Algorithm (Balancing Exploration and Exploitation)")
  plt.show()

  return sum(mu) / k  # Average reward per trial

# Example usage
k = 10  # Number of arms
T = 1000  # Number of trials

average_reward = thompson_sampling(k, T)
print(f"Average reward per trial: {average_reward:.4f}")

```

<img src="https://github.com/lanacstudio/RL_homework/blob/main/HW3/result/Thompson.png" width="90%"/>

**(4) 結果解釋：**

**空間上：**

 * Thompson Sampling通過考慮估計獎勵的不確定性來探索臂。
 * 具有較高不確定性（較高的標準差）的臂更有可能被選擇，因為算法尋求減少這種不確定性。

**時間上：**

 * Thompson Sampling的探索和利用策略在時間上也會演變。
 * 初始階段，由於算法需要評估每個臂的不確定性，探索的程度較高。這是因為在探索過程中，Thompson Sampling將更多的機會給予估計不確定性較高的臂，以期望進一步減少不確定性。
 * 隨著時間的推移，隨著更多的數據被收集，Thompson Sampling會更傾向於利用那些已經被證明為有利的臂，以獲取更多的獎勵。
 * 最終，隨著算法不斷學習，探索的程度會減少，而利用的程度會增加，從而導致平均獎勵的穩定或增加。
