import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from td_flow.config import (
    BackboneConfig,
    ModelConfig,
    TrainConfig,
    resolve_paper_max_steps,
)
from td_flow.model import TD2CFMModel
from td_flow.module import build_training_module


class ModelTest(unittest.TestCase):
    def test_identity_observation_encoder_uses_identity_for_vector_observations(self) -> None:
        config = ModelConfig(
            observation_shape=(4,),
            action_dim=2,
            backbone=BackboneConfig(kind="mlp", hidden_dims=(16,)),
            observation_encoder="identity",
        )
        model = TD2CFMModel(config)
        observation = torch.randn(3, 4)
        encoded = model.encode_observation(observation)

        self.assertEqual(model.latent_dim, 4)
        self.assertTrue(torch.allclose(encoded, observation))

    def test_policy_mode_controls_default_observation_encoder_behavior(self) -> None:
        single_policy = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="single_policy",
                observation_encoder="auto",
            )
        )
        multi_policy = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="multi_policy",
                observation_encoder="auto",
            )
        )

        self.assertTrue(single_policy.use_identity_encoder)
        self.assertFalse(multi_policy.use_identity_encoder)

    def test_multi_policy_model_requires_policy_embedding(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="multi_policy",
                observation_encoder="learned",
                policy_embedding_dim=3,
            )
        )
        batch = {
            "obs": torch.randn(2, 4),
            "next_obs": torch.randn(2, 4),
            "action": torch.randn(2, 2),
            "next_action": torch.randn(2, 2),
        }
        with self.assertRaises(ValueError):
            model.compute_state(batch, stage="fit")

    def test_paper_network_variant_uses_paper_widths(self) -> None:
        single_policy = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="single_policy",
                network_variant="paper",
                observation_encoder="identity",
            )
        )
        multi_policy = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="multi_policy",
                network_variant="paper",
                observation_encoder="learned",
                policy_embedding_dim=3,
            )
        )

        self.assertEqual(single_policy.context_encoder.network[0].out_features, 512)
        self.assertEqual(single_policy.vector_field.down_blocks[0].input_proj.out_features, 512)
        self.assertEqual(multi_policy.context_encoder.network[0].out_features, 1024)
        self.assertEqual(multi_policy.vector_field.down_blocks[0].input_proj.out_features, 1024)

        single_linear_layers = [
            module for module in single_policy.context_encoder.network if isinstance(module, nn.Linear)
        ]
        multi_linear_layers = [
            module for module in multi_policy.context_encoder.network if isinstance(module, nn.Linear)
        ]
        time_linear_layers = [
            module for module in single_policy.vector_field.time_encoder if isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(single_linear_layers), 3)
        self.assertEqual(len(multi_linear_layers), 3)
        self.assertEqual(len(time_linear_layers), 2)

    def test_paper_train_semantics_uses_step_interval(self) -> None:
        module = build_training_module(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
            ),
            TrainConfig(train_semantics="paper"),
        )
        self.assertEqual(module.optim["interval"], "step")

    def test_train_batch_end_updates_targets(self) -> None:
        module = build_training_module(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
            ),
            TrainConfig(train_semantics="paper"),
        )
        module.log = MagicMock()
        outputs = {
            "loss": torch.tensor(1.0),
            "loss_direct": torch.tensor(0.5),
            "loss_bootstrap": torch.tensor(0.25),
        }
        batch = {"obs": torch.zeros(3, 4)}

        with patch.object(module.td2_cfm, "update_targets") as update_targets:
            module.on_train_batch_end(outputs, batch, 0)

        update_targets.assert_called_once_with()

    def test_paper_defaults_split_by_policy_mode(self) -> None:
        single_policy_model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="single_policy",
                polyak=None,
            )
        )
        multi_policy_model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="multi_policy",
                policy_embedding_dim=3,
                polyak=None,
            )
        )
        single_module = build_training_module(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="single_policy",
            ),
            TrainConfig(train_semantics="paper", weight_decay=None),
        )
        multi_module = build_training_module(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                policy_mode="multi_policy",
                policy_embedding_dim=3,
            ),
            TrainConfig(train_semantics="paper", weight_decay=None),
        )

        self.assertEqual(single_policy_model.target_polyak, 0.999)
        self.assertEqual(multi_policy_model.target_polyak, 0.9999)
        self.assertEqual(single_module.optim["optimizer"]["weight_decay"], 1e-3)
        self.assertEqual(multi_module.optim["optimizer"]["weight_decay"], 1e-2)

    def test_paper_max_steps_split_by_policy_mode(self) -> None:
        self.assertEqual(resolve_paper_max_steps("single_policy"), 3_000_000)
        self.assertEqual(resolve_paper_max_steps("multi_policy"), 8_000_000)

    def test_loss_weights_default_to_gamma_split(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                gamma=0.95,
            )
        )

        direct_weight, bootstrap_weight = model.loss_weights()

        self.assertAlmostEqual(direct_weight, 0.05)
        self.assertAlmostEqual(bootstrap_weight, 0.95)

    def test_loss_weights_allow_explicit_override(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                gamma=0.99,
                direct_loss_weight=0.1,
                bootstrap_loss_weight=0.9,
            )
        )

        direct_weight, bootstrap_weight = model.loss_weights()

        self.assertAlmostEqual(direct_weight, 0.1)
        self.assertAlmostEqual(bootstrap_weight, 0.9)

    def test_loss_weights_direct_warmup_linear_schedule(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                gamma=0.99,
                loss_weight_schedule="direct_warmup_linear",
                loss_weight_warmup_steps=10,
                loss_weight_ramp_steps=20,
            )
        )

        model.set_loss_weight_step(0)
        self.assertEqual(model.loss_weights(), (1.0, 0.0))

        model.set_loss_weight_step(10)
        self.assertEqual(model.loss_weights(), (1.0, 0.0))

        model.set_loss_weight_step(20)
        direct_weight, bootstrap_weight = model.loss_weights()
        self.assertAlmostEqual(direct_weight, 0.505)
        self.assertAlmostEqual(bootstrap_weight, 0.495)

        model.set_loss_weight_step(30)
        direct_weight, bootstrap_weight = model.loss_weights()
        self.assertAlmostEqual(direct_weight, 0.01)
        self.assertAlmostEqual(bootstrap_weight, 0.99)

    def test_loss_weights_require_both_override_values(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                direct_loss_weight=0.1,
            )
        )

        with self.assertRaises(ValueError):
            model.loss_weights()

    def test_one_step_prediction_loss_contributes_to_total_loss(self) -> None:
        batch = {
            "obs": torch.randn(4, 4),
            "next_obs": torch.randn(4, 4),
            "action": torch.randn(4, 2),
            "next_action": torch.randn(4, 2),
        }

        torch.manual_seed(0)
        model_without_aux = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                one_step_prediction_loss_weight=0.0,
            )
        )
        torch.manual_seed(123)
        state_without_aux = model_without_aux.compute_state(batch, stage="fit")

        torch.manual_seed(0)
        model_with_aux = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                one_step_prediction_loss_weight=0.5,
            )
        )
        torch.manual_seed(123)
        state_with_aux = model_with_aux.compute_state(batch, stage="fit")

        expected_loss = state_without_aux["loss"] + 0.5 * state_with_aux["loss_one_step_prediction"]
        self.assertTrue(torch.allclose(state_with_aux["loss"], expected_loss, atol=1e-6))

    def test_one_step_prediction_uses_action_when_conditioning_is_enabled(self) -> None:
        torch.manual_seed(0)
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                one_step_prediction_loss_weight=1.0,
            )
        )

        state_latent = torch.randn(5, 4)
        action_a = torch.randn(5, 2)
        action_b = torch.randn(5, 2)

        prediction_a = model.predict_one_step_latent(state_latent, action_a)
        prediction_b = model.predict_one_step_latent(state_latent, action_b)

        self.assertFalse(torch.allclose(prediction_a, prediction_b))

    def test_orthogonal_initialization_zeroes_linear_biases(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                initialization="orthogonal",
            )
        )

        linear_modules = [module for module in model.vector_field.modules() if isinstance(module, nn.Linear)]
        self.assertTrue(linear_modules)
        for module in linear_modules:
            if module.bias is not None:
                self.assertTrue(torch.allclose(module.bias, torch.zeros_like(module.bias)))

    def test_late_mixture_bootstrap_time_sampling_biases_toward_endpoint(self) -> None:
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                bootstrap_time_sampling="late_mixture",
                bootstrap_time_late_prob=0.5,
                bootstrap_time_late_start=0.9,
            )
        )

        t = model.sample_bootstrap_time(20000, dtype=torch.float32)

        self.assertGreaterEqual(float((t >= 0.9).float().mean()), 0.45)
        self.assertGreaterEqual(float(t.min()), model.cfg.time_eps - 1e-7)

    def test_state_only_conditioning_ignores_current_action_in_velocity(self) -> None:
        torch.manual_seed(0)
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                state_only_conditioning=True,
            )
        )

        x_t = torch.randn(5, 4)
        t = torch.linspace(0.1, 0.9, 5)
        state_latent = torch.randn(5, 4)
        action_a = torch.randn(5, 2)
        action_b = torch.randn(5, 2)

        velocity_a = model.compute_velocity(x_t, t, state_latent, action_a)
        velocity_b = model.compute_velocity(x_t, t, state_latent, action_b)

        self.assertTrue(torch.allclose(velocity_a, velocity_b))

    def test_state_only_conditioning_ignores_next_action_in_bootstrap_target(self) -> None:
        torch.manual_seed(0)
        model = TD2CFMModel(
            ModelConfig(
                observation_shape=(4,),
                action_dim=2,
                observation_encoder="identity",
                state_only_conditioning=True,
            )
        )

        next_latent = torch.randn(6, 4)
        next_action_a = torch.randn(6, 2)
        next_action_b = torch.randn(6, 2)
        source = torch.randn(6, 4)
        t = torch.linspace(0.1, 0.9, 6)

        xt_a, target_a = model.bootstrap_target(next_latent, next_action_a, source, t)
        xt_b, target_b = model.bootstrap_target(next_latent, next_action_b, source, t)

        self.assertTrue(torch.allclose(xt_a, xt_b))
        self.assertTrue(torch.allclose(target_a, target_b))
        self.assertLessEqual(float(t.max()), 1.0 - model.cfg.time_eps + 1e-7)


if __name__ == "__main__":
    unittest.main()
