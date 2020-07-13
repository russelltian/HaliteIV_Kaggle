from train import utils, geometry
import unittest

class testHalite(unittest.TestCase):

    def test_load_replay(self):
        file = "replayJson.json"
        game = utils.Halite()
        game.load_replay(file)
        self.assertIsNotNone(game.replay)

    def test_load_data(self):
        file = "replayJson.json"
        game = utils.Halite()
        game.load_replay(file)
        game.load_data()
        self.assertIsNot(game.halite, [])
        self.assertIsNot(game.ship_actions, [])
        self.assertIsNot(game.shipyard_actions, [])

    def test_custom(self):
        file = "replayJson.json"
        game = utils.Halite()
        game.load_replay(file)
        game.load_data()
        length = len(game.halite)
        for i in range(length):
            print(game.ship_position[i])
            print(game.ship_actions[i], " \n")

    def test_find_winner(self):
        assert([1,2,3,4].index(max([1,2,3,4])) == 3)
        assert ([1, 5, 3, 4].index(max([1, 5, 3, 4])) == 1)
        assert ([1, 2, 12, 4].index(max([1, 2, 12, 4])) == 2)
        assert ([10, 2, 3, 4].index(max([10, 2, 3, 4])) == 0)
if __name__ == '__main__':
    unittest.main()

