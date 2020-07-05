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


if __name__ == '__main__':
    unittest.main()
    file = "replayJson.json"
    game = utils.Halite()
    game.load_replay(file)
    game.load_data()
