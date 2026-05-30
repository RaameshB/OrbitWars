import time
from kaggle_environments import make
import original_orbit_wars as ref

# Benchmark configuration
EPISODES = 50
STEPS_PER_EPISODE = 500
AGENT_COUNT = 4

print(f"Original Python Environment Benchmark")
print(f"Episodes: {EPISODES}")
print(f"Max Steps per Episode: {STEPS_PER_EPISODE}")

# Set up the environment
env = make("orbit_wars", debug=False)

def run_benchmark():
    total_steps_executed = 0
    start_time = time.time()

    for ep in range(EPISODES):
        # Reset the environment
        obs = env.reset(num_agents=AGENT_COUNT)
        state = obs[0].observation
        
        for step in range(STEPS_PER_EPISODE):
            if env.done:
                break
                
            # Use the random agent for all 4 players
            # In Kaggle environments, we provide a list of actions (one per player)
            actions = []
            for i in range(AGENT_COUNT):
                # The random agent needs to know which player it is
                # In the original env, observation for player i has player=i
                player_obs = obs[i].observation
                actions.append(ref.random_agent(player_obs))
                
            obs = env.step(actions)
            total_steps_executed += 1

    end_time = time.time()
    
    total_time = end_time - start_time
    steps_per_sec = total_steps_executed / total_time
    
    print(f"\n--- Results ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Environments Stepped: {total_steps_executed}")
    print(f"Steps / Second: {steps_per_sec:,.0f}")

if __name__ == "__main__":
    run_benchmark()
