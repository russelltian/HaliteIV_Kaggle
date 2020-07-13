from kaggle_environments import evaluate, make
import sys
from bot import v1_bot
from bot import t_bot
env = make("halite", configuration={"size": 5, "episodeSteps": 20}, debug=True)
trainer = env.train([None, "random"])
observation = trainer.reset()
mybot = None
turn = 0
while not env.done:
    if not mybot:
        mybot = v1_bot.Gameplay(observation, env.configuration)
        mybot.load_model()
    else:
        mybot.reset_board(observation, env.configuration)
    action = mybot.agent(observation, env.configuration)
    print("my action", action)
    print(mybot.board)
    observation = trainer.step(action)[0]
    print("Reward gained", observation.players[0][0])
    print("Turn", turn)
    turn +=1
out = env.render(mode="html", width=500, height=450)
f = open("replay.html", "wt")
f.write(out)
f.close()