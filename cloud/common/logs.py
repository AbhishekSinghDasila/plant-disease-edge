import logging

from common import settings


def setup():
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {settings.SERVICE} %(name)s %(levelname)s %(message)s")
