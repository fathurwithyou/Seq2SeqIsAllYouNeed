from __future__ import annotations

import numpy as np

from experiments.cnn.visualization import make_feature_grid, normalise_map, overlay_heatmap


def test_normalise_map_handles_constant_values():
    values = np.ones((3, 4), dtype=np.float32) * 7.0
    normalised = normalise_map(values)
    assert normalised.shape == values.shape
    assert np.all(normalised == 0.0)


def test_make_feature_grid_uses_single_example_and_channel_limit():
    feature_maps = np.arange(1 * 3 * 2 * 5, dtype=np.float32).reshape(1, 3, 2, 5)
    grid = make_feature_grid(feature_maps, max_channels=4, columns=2, pad=1)
    assert grid.shape == (7, 5)
    assert np.min(grid) >= 0.0
    assert np.max(grid) <= 1.0


def test_overlay_heatmap_returns_rgb_image_in_unit_range():
    image = np.zeros((4, 5, 3), dtype=np.float32)
    heatmap = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    overlay = overlay_heatmap(image, heatmap, alpha=0.5)
    assert overlay.shape == image.shape
    assert np.min(overlay) >= 0.0
    assert np.max(overlay) <= 1.0
