import unittest

import torch

from td_flow.config import BackboneConfig, ModelConfig
from td_flow.model import TD2CFMModel
from td_flow.planner import TD2CFMPlannerAdapter


class PlannerTest(unittest.TestCase):
    def test_planner_adapter_returns_batch_sample_costs(self) -> None:
        config = ModelConfig(
            observation_shape=(4,),
            action_dim=2,
            latent_dim=8,
            policy_embedding_dim=3,
            context_dim=8,
            backbone=BackboneConfig(kind="mlp", hidden_dims=(16,)),
            observation_encoder="learned",
            context_hidden_dims=(16,),
            vector_field_hidden_dims=(16,),
            ode_steps=2,
        )
        model = TD2CFMModel(config)
        adapter = TD2CFMPlannerAdapter(
            model,
            observation_key="pixels",
            goal_key="goal",
            policy_embedding_key="policy_embedding",
        )

        info = {
            "pixels": torch.randn(2, 3, 4),
            "goal": torch.randn(2, 3, 4),
            "policy_embedding": torch.randn(2, 3),
        }
        action_candidates = torch.randn(2, 3, 5, 2)
        costs = adapter.get_cost(info, action_candidates)
        self.assertEqual(costs.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
