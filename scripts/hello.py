import platform
import sys

from src.common.logging import get_logger


def main() -> None:
    logger = get_logger("day01")
    logger.info("Python version: %s", sys.version.split()[0])
    logger.info("Platform: %s", platform.platform())


if __name__ == "__main__":
    main()