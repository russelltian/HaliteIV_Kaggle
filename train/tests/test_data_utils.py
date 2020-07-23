from train import utils, geometry
import unittest

class testHalite(unittest.TestCase):

    # def test_load_replay(self):
    #     file = "replayJson.json"
    #     game = utils.Halite()
    #     game.load_replay(file)
    #     self.assertIsNotNone(game.replay)
    #
    # def test_load_data(self):
    #     file = "replayJson.json"
    #     game = utils.Halite()
    #     game.load_replay(file)
    #     game.load_data()
    #     self.assertIsNot(game.halite, [])
    #     self.assertIsNot(game.ship_actions, [])
    #     self.assertIsNot(game.shipyard_actions, [])
    #
    # def test_custom(self):
    #     file = "replayJson.json"
    #     game = utils.Halite()
    #     game.load_replay(file)
    #     game.load_data()
    #     length = len(game.halite)
    #     for i in range(length):
    #         print(game.ship_position[i])
    #         print(game.ship_actions[i], " \n")
    #
    def test_find_winner(self):
        assert([1, 2, 3, 4].index(max([1, 2, 3, 4])) == 3)
        assert ([1, 5, 3, 4].index(max([1, 5, 3, 4])) == 1)
        assert ([1, 2, 12, 4].index(max([1, 2, 12, 4])) == 2)
        assert ([10, 2, 3, 4].index(max([10, 2, 3, 4])) == 0)

    def test_load_replay_v2(self):
        file = "replayJson.json"
        game = utils.HaliteV2(file)
        self.assertIsNotNone(game.replay)
        self.assertIsNotNone(game.config)

    def test_load_total_turns(self):
        file = "replayJson.json"
        game = utils.HaliteV2(file)
        self.assertGreater(game.total_turns, 1)
        self.assertLessEqual(game.total_turns, 400)

    def test_convert_to_gameplay(self):
        file = "replayJson.json"
        game = utils.HaliteV2(file)
        gameplay = game.convert_to_game_play(2)
        self.assertIsNotNone(gameplay)

if __name__ == '__main__':
    unittest.main()

