import os

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.dirname(__file__))

LOGGER_CONFIG_FILE = os.path.join(PROJECT_ABSOLUTE_PATH, "config", "logging.conf")
LOGS_CATALOG_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "logs")
QUERIES_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "queries")
PICKLES_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "pickles")
FEATURES_STATS_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, "features_stats")
