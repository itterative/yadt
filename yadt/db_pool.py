import time
import sqlite3

from contextlib import contextmanager

from collections import deque
from threading import Condition, Semaphore, Thread

class Sqlite3DBPool:
    def __init__(self, database: str, default_timeout: float = 10, max_connections: int = 10, **pragmas: str):
        self._database = database
        self._max_connections = max_connections
        self._default_timeout = default_timeout
        self._pragmas = pragmas
        self._connection_sem_allowed = 0
        self._connection_sem = Semaphore(0)
        self._connection_count = 0
        self._connections_open = False
        self._connections_open_cv = Condition()
        self._connection_pool: deque[sqlite3.Connection] = deque()
        self._connections_active: set[sqlite3.Connection] = set()
        
        self._cleanup_thread = Thread(target=self._cleanup, daemon=True)
        self._cleanup_thread.start()

    @contextmanager
    def connection(self, timeout=None):
        if timeout is None:
            timeout = self._default_timeout

        timeout_t = time.time() + timeout

        with self._connections_open_cv:
            while not self._connections_open:
                if not self._connections_open_cv.wait(timeout=max(0, timeout_t-time.time())):
                    raise TimeoutError('Could not aquire database connection')

            if not self._connection_sem.acquire(timeout=max(0, timeout_t-time.time())):
                raise TimeoutError('Could not aquire database connection')
            
            try:
                connection = None

                if len(self._connection_pool) > 0:
                    connection = self._connection_pool.popleft()
                elif self._connection_count < self._max_connections:
                    connection = sqlite3.connect(self._database, check_same_thread=False)

                    for pragma, value in self._pragmas.items():
                        connection.execute(f'pragma {pragma} = {value}')

                    self._connection_count += 1

                assert connection is not None, "no connection was aquired"
                self._connections_active.add(connection)

                try:
                    yield connection
                    connection.commit()
                except sqlite3.Error:
                    connection.rollback()
            finally:
                if connection is not None:
                    self._connections_active.remove(connection)
                    self._connection_pool.append(connection)
                    self._connection_sem.release()

    def open(self):
        with self._connections_open_cv:
            while self._connection_sem_allowed < self._max_connections:
                self._connection_sem_allowed += 1
                self._connection_sem.release()

            self._connections_open = True
            self._connections_open_cv.notify_all()


    def close(self):
        self._connections_open = False

        with self._connections_open_cv:
            while self._connection_sem_allowed > 0:
                self._connection_sem_allowed -= 1
                self._connection_sem.acquire()

            self._connections_open_cv.notify_all()

        self._connection_sem.release()
        self._cleanup_connections(all=True)
        self._connection_sem.acquire()
        
    def _cleanup(self):
        import time

        try:
            while True:
                # TODO: connection idle time
                time.sleep(60)
                self._cleanup_connections()                
        except KeyboardInterrupt:
            return
    
    def _cleanup_connections(self, all: bool = False):
        cleaned_up_connections = 0

        try:
            while self._connection_count > 0:
                if not self._connection_sem.acquire(blocking=all):
                    break

                try:
                    if len(self._connection_pool) <= 0:
                        break

                    connection = self._connection_pool.popleft()
                    assert connection is not None, "no connection was aquired"

                    connection.close()
                    cleaned_up_connections += 1
                    self._connection_count -= 1
                finally:
                    self._connection_sem.release()
        finally:
            # print('cleaned_up_connections', cleaned_up_connections)
            pass
