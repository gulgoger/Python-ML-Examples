import numpy as np
import matplotlib.pyplot as plt

# Number of arms (actions)
num_arms = 5

# True mean reward for each arm (unknown to the agent)
true_means = np.random.normal(0, 1, num_arms)

# Number of time steps
num_steps = 1000

# Parameters for UCB
exploration_parameter = 2.0
num_selections = np.zeros(num_arms)
sum_rewards = np.zeros(num_arms)

# UCB function
def ucb(t, num_selections, sum_rewards):
    return [sum_rewards[i] / max(1, num_selections[i]) + exploration_parameter * np.sqrt(np.log(t + 1) / max(1, num_selections[i])) for i in range(num_arms)]

# Run UCB algorithm
for t in range(num_steps):
    ucb_values = ucb(t, num_selections, sum_rewards)
    chosen_arm = np.argmax(ucb_values)
    
    # Simulate pulling the chosen arm and observe the reward
    reward = np.random.normal(true_means[chosen_arm], 1)
    
    # Update statistics
    num_selections[chosen_arm] += 1
    sum_rewards[chosen_arm] += reward

# Display the true means and the estimated means using UCB
print("True means:", true_means)
print("Estimated means using UCB:", sum_rewards / num_selections)

# Plot the results
plt.bar(range(num_arms), true_means, color='blue', label='True Means', alpha=0.5)
plt.bar(range(num_arms), sum_rewards / num_selections, color='orange', label='Estimated Means (UCB)')
plt.xlabel('Arm')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()

