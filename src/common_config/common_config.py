import os

PROJECT_ABSOLUTE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir))

LOGGER_CONFIG_FILE = os.path.join(PROJECT_ABSOLUTE_PATH, "common_config", "logging.conf")
LOGS_CATALOG_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "logs")
QUERIES_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "queries")
PICKLES_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "pickles")
FEATURES_STATS_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "features_stats")
