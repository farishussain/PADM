# PADM - Planning and Decision Making

A comprehensive reinforcement learning project implementing Q-learning algorithms in custom OpenAI Gymnasium environments. This project simulates a student's journey through academic coursework using grid-world environments.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Assignments](#assignments)
- [Environment Details](#environment-details)
- [Algorithms](#algorithms)
- [Visualization](#visualization)
- [Results](#results)

## 🎯 Overview

This project demonstrates various concepts in **Planning and Decision Making** through reinforcement learning. The main focus is on implementing Q-learning algorithms in custom environments where an agent (student) must navigate through academic challenges to successfully complete a course.

### Learning Objectives
- Understand reinforcement learning fundamentals
- Implement Q-learning from scratch
- Create custom OpenAI Gymnasium environments
- Visualize agent behavior and learning progress
- Explore dynamic programming concepts

## 📁 Project Structure

```
PADM/
├── README.md                    # This file
├── dynamic_programming.py       # Fibonacci implementation with different approaches
├── week3_practicals.ipynb      # Introduction to Gymnasium environments
├── week4_practicals.py         # Custom environment development
├── q_table.npy                 # Saved Q-table from training
│
├── Assignment_1/               # Basic custom environment
│   ├── env_fah8265.py         # Student navigation environment
│   └── Images/                # Visual assets (student, exam, etc.)
│
├── Assignment_2/               # Value iteration implementation
│   ├── fah8265.py             # Grid world with danger states
│   └── fah8265.pdf            # Assignment documentation
│
├── Assignment_3/               # Advanced Q-learning
│   ├── main.py                # Main execution script
│   ├── padm_env.py            # Enhanced environment
│   ├── Q_learning.py          # Q-learning implementation
│   └── Coding/                # Additional implementations
│
├── Q_Learning/                 # Core Q-learning implementation
│   ├── main.py                # Training and visualization
│   ├── padm_env.py            # Environment definition
│   └── q_learning.py          # Q-learning algorithm
│
├── Lectures/                   # Course materials
└── Presentation/              # Final presentation materials
```

## ✨ Features

### Custom Environment (`PadmEnv`)
- **Grid-based navigation** (5x5 or 7x7 grids)
- **Visual rendering** with real images
- **Multi-objective tasks**: assignments, presentations, exams
- **Obstacle avoidance**: blockers that cause deregistration
- **Reward system**: task completion rewards and penalties

### Q-Learning Implementation
- **Epsilon-greedy exploration** strategy
- **Configurable hyperparameters**
- **Q-table persistence** (save/load functionality)
- **Training progress visualization**
- **Convergence analysis**

### Visualization Tools
- **Real-time environment rendering**
- **Q-value heatmaps** for all actions
- **Training progress plots**
- **Performance metrics tracking**

## 🚀 Installation

### Prerequisites
```bash
python >= 3.8
```

### Required Libraries
```bash
pip install gymnasium
pip install numpy
pip install matplotlib
pip install seaborn
pip install pygame
pip install tqdm
```

### Optional (for notebooks)
```bash
pip install jupyter
pip install ipython
```

## 🎮 Usage

### Basic Q-Learning Training

```bash
cd Q_Learning
python main.py
```

### Assignment 3 (Advanced Implementation)

```bash
cd Assignment_3
python main.py
```

### Custom Configuration

```python
from padm_env import create_env
from Q_learning import train_q_learning, visualize_q_table

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
no_episodes = 1000

# Environment setup
goal_coordinates = (4, 4)
hell_state_coordinates = [(2, 1), (0, 4)]

# Create and train
env = create_env(goal_coordinates, hell_state_coordinates)
train_q_learning(env, no_episodes, epsilon, epsilon_min, 
                epsilon_decay, learning_rate, gamma)

# Visualize results
visualize_q_table(hell_state_coordinates, goal_coordinates)
```

## 📚 Assignments

### Assignment 1: Basic Environment
- **Objective**: Create a custom Gymnasium environment
- **Features**: Student navigation, task completion, visual rendering
- **Key Concepts**: Environment design, reward engineering

### Assignment 2: Value Iteration
- **Objective**: Implement value iteration for grid world
- **Features**: Danger states, living rewards, transition probabilities
- **Key Concepts**: Dynamic programming, Bellman equations

### Assignment 3: Advanced Q-Learning
- **Objective**: Comprehensive Q-learning implementation
- **Features**: Configurable environments, enhanced visualization
- **Key Concepts**: Reinforcement learning, exploration vs exploitation

## 🌍 Environment Details

### State Space
- **Grid positions**: (x, y) coordinates
- **Task status**: Completed assignments and presentations
- **Goal proximity**: Distance to exam location

### Action Space
- **0**: Move Up
- **1**: Move Down  
- **2**: Move Right
- **3**: Move Left

### Reward Structure
- **Task completion**: +10 to +30 points
- **Goal achievement**: +50 points (with all tasks completed)
- **Blocker collision**: -1 to -10 points
- **Boundary violation**: -1000 points

### Visual Elements
- 🧑‍🎓 **Student**: Agent representation
- 📝 **Assignments**: Task locations
- 🎤 **Presentation**: Special task
- 🎓 **Exam**: Goal state
- ❌ **Cancel**: Blocker/hell states

## 🤖 Algorithms

### Q-Learning
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Parameters**:
- Learning rate (α): 0.01
- Discount factor (γ): 0.99
- Exploration rate (ε): 1.0 → 0.1 (decay: 0.995)

### Epsilon-Greedy Strategy
```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q_table[state])  # Exploit
```

### Dynamic Programming
- **Memoization**: Store computed values to avoid recomputation
- **Bottom-up**: Build solutions from base cases
- **Applications**: Fibonacci series, value iteration

## 📊 Visualization

### Q-Value Heatmaps
- **Action-specific Q-values** displayed as heatmaps
- **Color coding**: Higher values = warmer colors
- **State analysis**: Understanding agent preferences

### Training Progress
- **Episode rewards** over time
- **Convergence analysis**
- **Exploration vs exploitation balance**

### Environment Rendering
- **Real-time visualization** during training
- **Agent movement tracking**
- **Task completion status**

## 📈 Results

### Performance Metrics
- **Success rate**: Percentage of successful episodes
- **Average reward**: Mean reward per episode
- **Convergence time**: Episodes to reach stable policy
- **Exploration efficiency**: Balance between exploration and exploitation

### Typical Training Results
- **Episodes to convergence**: 500-1000
- **Final success rate**: 85-95%
- **Optimal policy**: Direct path to goal with task completion

## 🔧 Configuration Options

### Environment Parameters
```python
grid_size = 5                    # Grid dimensions
goal_coordinates = (4, 4)        # Exam location
hell_state_coordinates = [(2,1)] # Blocker positions
```

### Training Parameters
```python
learning_rate = 0.01        # Step size for Q-value updates
gamma = 0.99               # Discount factor for future rewards
epsilon = 1.0              # Initial exploration rate
epsilon_decay = 0.995      # Exploration decay rate
no_episodes = 1000         # Number of training episodes
```

## 🤝 Contributing

This project is part of academic coursework. For improvements or questions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes as part of the PADM course curriculum.

## 👨‍💻 Author

**Faris Hussain**  
Student ID: fah8265  
Course: Planning and Decision Making (PADM)  
Institution: Technische Hochschule Ingolstadt

---

*This README provides a comprehensive guide to understanding and using the PADM reinforcement learning project. For detailed implementation specifics, refer to the individual source files and assignments.*
