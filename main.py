from kaggle_environments import make
env = make("connectx", {rows: 10, columns: 8, inarow: 5})

def agent(observation, configuration):
  print(observation) # {board: [...], mark: 1}
  print(configuration) # {rows: 10, columns: 8, inarow: 5}
  return 3 # Action: always place a mark in the 3rd column.

# Run an episode using the agent above vs the default random agent.
env.run([agent, "random"])

# Print schemas from the specification.
print(env.specification.observation)
print(env.specification.configuration)
print(env.specification.action)