import unittest

import numpy as np
import torch

from src.models import FLEMProtoNet
from src.training import train_episode
from src.utils.metrics import evaluate_episodic_predictions


class EpisodeProtocolTests(unittest.TestCase):
    def test_independent_episode_label_permutation(self):
        labels = [np.array([[1, 0], [0, 1], [1, 0]]), np.array([[0, 1], [0, 1], [1, 0]])]
        scores = [np.array([[.8, .2], [.3, .6], [.4, .3]]), np.array([[.3, .4], [.6, .7], [.8, .1]])]
        expected = evaluate_episodic_predictions(labels, scores, threshold_search=True)
        actual = evaluate_episodic_predictions([labels[0], labels[1][:, ::-1]],
                                               [scores[0], scores[1][:, ::-1]], threshold_search=True)
        for name in expected:
            self.assertAlmostEqual(expected[name], actual[name], places=12, msg=name)

    def test_no_test_threshold_search_and_missing_positive_coverage(self):
        result = evaluate_episodic_predictions([np.array([[1, 0], [0, 0]])],
                                              [np.array([[.4, .1], [.2, .1]])], threshold=.3)
        self.assertEqual(result['AP-valid-label-fraction'], .5)
        self.assertNotIn('Best-threshold', result)
        self.assertEqual(result['Micro-F1'], 1.)

    def test_query_gradient_only_selected_labels(self):
        torch.manual_seed(42)
        model = FLEMProtoNet(num_features=8, num_labels=4, input_dim=8,
                            input_is_feature=True, label_weight_mode='binary')
        selected = torch.tensor([3, 1])
        outputs, _ = train_episode(model, torch.optim.SGD(model.parameters(), lr=.01),
                                   torch.randn(8, 8), torch.eye(4).repeat(2, 1),
                                   torch.randn(4, 8), torch.eye(4), episode_labels=selected)
        self.assertEqual(tuple(outputs['logits'].shape), (4, 2))
        self.assertTrue(torch.equal(model.bias.grad[[0, 2]], torch.zeros(2)))
        self.assertGreater(float(model.bias.grad[[1, 3]].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
