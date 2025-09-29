import queue


class SearchResultManager:
    def __init__(self):
        self.result_queue = queue.Queue()

    def add_result(self, result):
        self.result_queue.put(result)

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


search_result_manager = SearchResultManager()
