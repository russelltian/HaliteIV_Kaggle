from kaggle_environments import evaluate, make
from bot import vae_bot
<<<<<<< HEAD
env = make("halite", configuration={"size": 21, "episodeSteps": 75}, debug=True)
=======
env = make("halite", configuration={"size": 21, "episodeSteps": 100}, debug=True)
>>>>>>> e5bf5d7ca6ab9b9b89b85c0da762c0082cb567fd
trainer = env.train([None, "random", "random", "random"])
observation = trainer.reset()
mybot = None
turn = 0
while not env.done:
    if not mybot:
        mybot = vae_bot.VaeBot(observation, env.configuration)
    else:
        mybot.reset_board(observation, env.configuration)
    action = mybot.agent(observation, env.configuration)
    print("my action", action)
    # print(mybot.board)
    observation = trainer.step(action)[0]
    print("Reward gained", observation.players[0][0])
    print("Turn", turn)
    turn += 1
out = env.render(mode="html", width=500, height=450)
f = open("replay.html", "wt")
f.write(out)
f.close()