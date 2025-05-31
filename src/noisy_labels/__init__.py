import os
import sys

from loguru import logger
from tqdm.auto import tqdm

logger.remove()
# logger.add(
#     sys.stdout,
#     colorize=True,
#     format="<green>{time:HH:mm:ss}</green> | "
#     "<level>{level: <8}</level> | "
#     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
#     "<level>{message}</level>",
# )


class TqdmCompatibleSink:
    def write(self, message):
        tqdm.write(message.rstrip())  # tqdm.write handles newline


logger.add(
    TqdmCompatibleSink(),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


PATH = os.getcwd()


def main():
    logger.info("ciao")
