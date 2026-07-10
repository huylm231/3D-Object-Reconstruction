import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_uv_mapping_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "image_processing" / "13_uv_mapping.py"
    spec = importlib.util.spec_from_file_location("uv_mapping_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VertexColorRemappingTest(unittest.TestCase):
    def test_remap_vertex_colors_to_vertices_uses_nearest_neighbor(self):
        module = load_uv_mapping_module()

        src_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        dst_vertices = np.array(
            [[0.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        vertex_colors = np.array(
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            dtype=np.uint8,
        )

        remapped = module._remap_vertex_colors_to_vertices(
            src_vertices, dst_vertices, vertex_colors
        )

        np.testing.assert_array_equal(remapped[0], vertex_colors[0])
        np.testing.assert_array_equal(remapped[1], vertex_colors[2])
        np.testing.assert_array_equal(remapped[2], vertex_colors[0])


if __name__ == "__main__":
    unittest.main()
