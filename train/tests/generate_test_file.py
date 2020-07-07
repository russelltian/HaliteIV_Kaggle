from kaggle_environments import evaluate, make
import sys
sys.path.append("../")

env = make("halite", configuration={"size": 5, "episodeSteps": 10}, debug=True)
env.run(["random", "random"])
out = env.render(mode="html", width=500, height=450)
f = open("replay.html", "w")
f.write(out)
f.close()

out = env.render(mode="json")
f = open("replayJson.json","w")
f.write(str(out))
f.close()