from classes.query_loader import QueryLoader


class ConfigQueryLoader(QueryLoader):
    def __init__(self, config):
        self.config = config

    def load_query(self, query_name):
        return self.config["sql_queries"][query_name]
