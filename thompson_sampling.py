import numpy as np
import matplotlib.pyplot as plt

# Number of arms (actions)
num_arms = 5

# True mean reward for each arm (unknown to the agent)
true_means = np.random.normal(0, 1, num_arms)

# Number of time steps
num_steps = 1000

# Parameters for Thompson Sampling
alpha = np.ones(num_arms)  # Prior parameters (initialized to 1 for a uniform distribution)
beta = np.ones(num_arms)

# Thompson Sampling function
def thompson_sampling(alpha, beta):
    return [np.random.beta(alpha[i], beta[i]) for i in range(num_arms)]

# Run Thompson Sampling algorithm
for t in range(num_steps):
    sampled_means = thompson_sampling(alpha, beta)
    chosen_arm = np.argmax(sampled_means)
    
    # Simulate pulling the chosen arm and observe the reward
    reward = np.random.normal(true_means[chosen_arm], 1)
    
    # Update parameters using Bayesian update
    if reward == 1:
        alpha[chosen_arm] += 1
    else:
        beta[chosen_arm] += 1

# Display the true means and the estimated means using Thompson Sampling
print("True means:", true_means)
print("Estimated means using Thompson Sampling:", alpha / (alpha + beta))

# Plot the results
plt.bar(range(num_arms), true_means, color='blue', label='True Means', alpha=0.5)
plt.bar(range(num_arms), alpha / (alpha + beta), color='green', label='Estimated Means (Thompson Sampling)')
plt.xlabel('Arm')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()

