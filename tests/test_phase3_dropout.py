import unittest

import torch

from dual_agent.training.loops import disable_training_dropout


class Phase3DropoutTests(unittest.TestCase):
    def test_attention_matches_evaluation_and_keeps_gradients(self):
        torch.manual_seed(7)
        model = torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0.5, batch_first=True)
        x = torch.randn(3, 4, 8, requires_grad=True)
        disable_training_dropout(model)
        model.train()
        first, second = model(x), model(x)
        model.eval()
        evaluated = model(x)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, evaluated)
        first.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_nested_dropout_and_recurrent_dropout_are_disabled(self):
        model = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Dropout(0.5)),
            torch.nn.GRU(8, 8, num_layers=2, dropout=0.5),
        ])
        disable_training_dropout(model)
        self.assertEqual(model[0][0].p, 0.)
        self.assertEqual(model[1].dropout, 0.)

    def test_config_exposes_phase3_adaptation_controls(self):
        from dual_agent.config import load_config
        config = load_config("configs/ieee13_phase3_recommended.yaml")
        self.assertTrue(config.training.phase3_disable_dropout)
        self.assertTrue(config.training.phase3_forecaster_first)
        self.assertEqual(config.training.phase3_validation_fraction, 0.2)


if __name__ == '__main__':
    unittest.main()
