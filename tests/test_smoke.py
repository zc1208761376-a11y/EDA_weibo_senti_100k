import os
import sys

sys.path.append(os.path.abspath("src"))

from common.logging import get_logger


def test_get_logger() -> None:
    logger = get_logger("test")
    assert logger.name == "test"