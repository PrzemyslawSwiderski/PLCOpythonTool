from abc import abstractmethod


class QueryLoader:
    @abstractmethod
    def load_query(self, query_name):
        pass
