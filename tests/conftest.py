from pathlib import Path
import pytest
from dataset.synthetic import generate


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    images = generate(root)
    return root, images
