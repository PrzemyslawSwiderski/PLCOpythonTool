class QueryLoader:
    def load_query(self, query_name):
        with open(query_name, 'r') as query_file:
            return query_file.read()
