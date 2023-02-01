from abc import ABC, abstractmethod
from typing import Optional, Any, List
import time


class LanguageModelInterface(ABC):

    def __init__(self, key: str = None, limit: Optional[int] = None, dry_run: bool = False):
        """
        :param key:
        :param limit: limit interface to N request per minute
        :param dry_run:
        """
        self.dry_run = dry_run
        self._limit = limit
        self._min_request_time = None if self._limit is None else 60.0 / self._limit
        self._last_request_time = None

    def _await_limit(self):
        if self._limit is None:
            return
        if self._last_request_time is None:
            self._last_request_time = time.perf_counter()
            return

        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_request_time
        remaining_time = self._min_request_time - elapsed_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)
        self._last_request_time = time.perf_counter()

    def query(self, query_text: str, log_info: Any = "") -> Optional[str]:
        print(f"[querying{'' if log_info=='' else ' | '}{log_info}]", query_text)
        if self.dry_run:
            return None
        self._await_limit()
        answer = self.do_query(query_text)
        return answer

    def query_embedding(self, query_text: str, log_info: Any = "") -> Optional[list]:
        print(f"[querying{'' if log_info=='' else ' | '}{log_info}]", query_text)
        if self.dry_run:
            return None
        self._await_limit()
        answer = self.do_get_embedding(query_text)
        return answer

    def query_embedding_batch(self, query_texts: List[str], log_info: Any = "") -> Optional[list]:
        print(f"[querying batch{'' if log_info == '' else ' | '}{log_info}]")
        for q in query_texts:
            print(">", q)

        if self.dry_run:
            return None
        self._await_limit()
        answer = self.do_get_embedding_batch(query_texts)
        return answer

    @abstractmethod
    def do_query(self, query_text: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def do_get_embedding(self, query_text: str) -> list:
        raise NotImplementedError()

    @abstractmethod
    def do_get_embedding_batch(self, query_text: List[str]) -> list:
        raise NotImplementedError()
