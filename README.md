# Classifier Agent For Pacman

This project introduces a **K-Nearest Neighbors (KNN) Classifier Agent for Pacman**. Using supervised learning, this classifier agent selects the optimal actions for Pacman by analyzing rewards associated with similar game states. The KNN classifier utilizes customized feature weights to adjust its behavior dynamically based on the presence of ghosts, food, walls, and visibility in the Pacman grid environment.

## Features

- **KNN-Based Decision Making**: The agent uses KNN to predict Pacman’s next action based on similarities to past states.
- **Feature Weight Adjustments**:  Weighted parameters for ghosts, visible ghosts, food, and walls allow customization of the classifier’s responsiveness to different environmental features.
- **Dynamic k Selection**: The agent determines the optimal k value to improve prediction accuracy.
- **Pacman Simulation**: This agent is compatible with the Pacman simulation environment and can be used with various layouts and configurations of the game.

  
## Prerequisites

- **Python 3** (or Python 3.x if applicable)
- **Pacman AI Project**: Make sure you have the full Pacman project environment.
- **NumPy**: For array operations and distance calculations
- **scikit-learn**: For train_test_split and KNN modeling

Install Dependencies With:
```bash
pip install numpy scikit-learn
```
  
## Running the Agent

To run the **ClassifierAgent** in the Pacman environment, follow these steps:

1. **Navigate to the Project Directory**: First, ensure you're in the correct directory where the Pacman code is located:

```bash
cd pacman  # Navigate to the directory containing pacman.py
```

2. **Run the Agent**: Use the following command:

```bash
python3 pacman.py --pacman ClassifierAgent
```

### Explanation of Command:
- `ClassifierAgent`: Use the Classifier-based agent for decision-making.
- `--pacman`: Instructs pacman.py to search for and execute the specified agent (ClassifierAgent in this case) during gameplay.

## How the Classifier Agent Works

- **State Representation**: Each state represents a unique configuration of Pacman, the ghosts, and the grid.
- **Feature-Based Rewards:**: The agent uses a weighted scoring system to adjust its focus on specific features such as food, ghosts, and visible threats. Rewards are assigned to incentivize beneficial actions (like eating food or avoiding ghosts) and guide Pacman towards favorable states.
- **Nearest-Neighbor Classification**: Using the K-Nearest Neighbors (KNN) algorithm, the agent compares the current state to past states to determine the best action based on similar historical data. The distance to these neighbors (similar states) is calculated, with the agent choosing the action associated with the most frequent or closest neighbors.
- **Policy Execution**: Once an optimal action is determined from the nearest neighbors, the agent executes this action within the game, continuously updating its behavior as it learns from new state transitions and outcomes.

## Future Improvements

- **Ghost Avoidance**: Implement improved ghost prediction strategies for better ghost evasion.
- **Dynamic Feature Weighting**: Allow weights to adapt in real-time based on gameplay feedback.
- **Additional Algorithms**: Implement other supervised learning algorithms, like decision trees or SVM, for comparison.

## License
This project is licensed for educational purposes under the following conditions:

- You may not distribute or publish solutions to these projects.
- You must retain this notice in all copies or substantial portions of the code.
- You must provide clear attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

For detailed information, refer to the [LICENSE](LICENSE) file.