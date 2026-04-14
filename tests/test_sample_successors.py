import unittest

import numpy as np

from td_flow.sample_successors import (
    _compute_uncertainty_scores,
    _make_image_grid,
    _sort_indices_by_uncertainty,
)


class SampleSuccessorsTest(unittest.TestCase):
    def test_make_image_grid_uses_requested_columns(self) -> None:
        frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(5)]
        grid = _make_image_grid(frames, cols=3, padding=2, label_tiles=False)
        self.assertEqual(grid.size, (3 * 5 + 4 * 2, 2 * 4 + 3 * 2))

    def test_compute_uncertainty_scores_ranks_outlier_higher(self) -> None:
        predictions = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [5.0, 5.0],
            ],
            dtype=np.float32,
        )
        raw_scores, percentiles = _compute_uncertainty_scores(predictions)
        self.assertEqual(raw_scores.shape, (4,))
        self.assertEqual(percentiles.shape, (4,))
        self.assertGreater(raw_scores[-1], raw_scores[0])
        self.assertEqual(float(percentiles[-1]), 1.0)

    def test_make_image_grid_accepts_tile_labels(self) -> None:
        frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(2)]
        grid = _make_image_grid(
            frames,
            cols=2,
            padding=1,
            label_tiles=True,
            tile_labels=["000 u=0.10", "001 u=0.90"],
        )
        self.assertEqual(grid.size, (2 * 5 + 3 * 1, 1 * (4 + 18) + 2 * 1))

    def test_sort_indices_by_uncertainty_orders_low_to_high(self) -> None:
        percentiles = np.array([0.8, 0.1, 0.5], dtype=np.float32)
        order = _sort_indices_by_uncertainty(percentiles, descending=False)
        np.testing.assert_array_equal(order, np.array([1, 2, 0]))

    def test_sort_indices_by_uncertainty_orders_high_to_low(self) -> None:
        percentiles = np.array([0.8, 0.1, 0.5], dtype=np.float32)
        order = _sort_indices_by_uncertainty(percentiles, descending=True)
        np.testing.assert_array_equal(order, np.array([0, 2, 1]))


if __name__ == "__main__":
    unittest.main()
