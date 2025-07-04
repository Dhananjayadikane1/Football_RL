# ⚽️ 2D Football Penalty Reinforcement Learning

This project simulates a 2D penalty shootout using Reinforcement Learning. It features:
- A **Striker agent** using Dueling Double DQN
- A **Goalkeeper agent** using PPO
- A custom football environment with angle and reaction modeling

## 📁 Project Structure

```
football-rl/
├── agents/
│   ├── dqn_striker.py          # Dueling Double DQN striker agent
│   ├── ppo_goalkeeper.py       # PPO goalkeeper agent
│   └── random_goalkeeper.py    # Optional random keeper baseline
│
├── env/
│   └── football_env.py         # Custom 2D football environment
│
├── train/
│   ├── train_dqn_striker.py    # Train only the striker
│   ├── train_ppo_keeper.py     # Train only the goalkeeper
│   └── train_dual_agent.py     # Train both striker and keeper
│
├── visualize/
│   ├── game_visualizer.py      # Plot single penalty snapshot
│   └── plot_learning_curve.py  # Plot training curves
│
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point (optional visual run)
├── README.md                   # This file
└── .vscode/settings.json       # VS Code config (optional)
```

---
---

## ▶️ How to Run

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Train Agents

```bash
# Train striker only (Dueling Double DQN)
python train/train_dqn_striker.py
```

```bash
# Train keeper only (PPO)
python train/train_ppo_keeper.py
```

```bash
# Train both agents together
python train/train_dual_agent.py
```

---

## 📈 Sample Output

```
Ep 500/500 | S: 1.00 | K: -1.00 | Result: GOAL  
Ep 1000/1000 | S: -1.00 | K: 1.00 | Result: SAVE
```

---

## 🧮 Degrees of Freedom (DoF)

- **Ball**: `[x, y]`  
- **Keeper**: x-axis movement only  
- **Environment includes**:  
  - Shot angle  
  - Distance to goal  
  - Keeper's reaction delay

---

## 📌 Key Highlights

✅ Custom 2D environment  
✅ Striker with Dueling DQN using shot angle  
✅ Keeper with PPO & advantage estimation  
✅ Dual-agent training with visualizations  
✅ Portfolio-ready reinforcement learning project

---

## 🧑‍💻 Author

**Dhananjay Adikane**  
M.Tech in Data Science, DIAT Pune  
GitHub: [@Dhananjayadikane1](https://github.com/Dhananjayadikane1)

---

## 📜 License

This project is open-source under the [MIT License](LICENSE)
