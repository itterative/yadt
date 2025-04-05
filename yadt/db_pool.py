import time
import sqlite3

from contextlib import contextmanager
from dataclasses import dataclass

from collections import deque
from threading import Condition, Semaphore, Thread, Lock

@dataclass
class PoolConnection:
    last_used: float
    connection: sqlite3.Connection

class Sqlite3DBPool:
    def __init__(self, database: str, default_timeout: float = 10, idle_timeout: float = 30, max_connections: int = 10, **pragmas: str):
        self._database = database
        self._max_connections = max_connections
        self._idle_timeout = idle_timeout
        self._default_timeout = default_timeout
        self._pragmas = pragmas
        self._connection_sem_allowed = 0
        self._connection_sem = Semaphore(0)
        self._connection_count = 0
        self._connections_open = False
        self._connections_open_cv = Condition()
        self._connection_pool: deque[PoolConnection] = deque()
        self._connection_pool_lock = Lock()
        self._connection_cleanup_thread: Thread = None

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
            pool_item = None

            with self._connection_pool_lock:
                if len(self._connection_pool) > 0:
                    pool_item = self._connection_pool.popleft()
                elif self._connection_count < self._max_connections:
                    connection = sqlite3.connect(self._database, check_same_thread=False)

                    for pragma, value in self._pragmas.items():
                        connection.execute(f'pragma {pragma} = {value}')

                    pool_item = PoolConnection(last_used=time.time(), connection=connection)
                    self._connection_count += 1

            try:
                yield pool_item.connection
                pool_item.connection.commit()
            except sqlite3.Error:
                pool_item.connection.rollback()
        finally:
            if pool_item is not None:
                pool_item.last_used = time.time()

                self._connection_pool.append(pool_item)
                self._connection_sem.release()

    def open(self):
        with self._connections_open_cv:
            while self._connection_sem_allowed < self._max_connections:
                self._connection_sem_allowed += 1
                self._connection_sem.release()

            self._connections_open = True
            self._connections_open_cv.notify_all()

        if self._connection_cleanup_thread is None:
            self._connection_cleanup_thread = Thread(name=f'Sqlite3DBPool._cleanup', target=self._cleanup, kwargs=dict(until_closed=True), daemon=True)
            self._connection_cleanup_thread.start()

    def close(self):
        with self._connections_open_cv:
            self._connections_open = False

            while self._connection_sem_allowed > 0:
                self._connection_sem_allowed -= 1
                self._connection_sem.acquire()

            self._connections_open_cv.notify_all()

        self._connection_sem.release()
        self._cleanup_connections(all=True)
        self._connection_sem.acquire()

        if self._connection_cleanup_thread is not None:
            self._connection_cleanup_thread.join()
            self._connection_cleanup_thread = None
        
    def _cleanup(self, until_closed: bool = False):
        try:
            while until_closed:
                with self._connections_open_cv:
                    self._connections_open_cv.wait(timeout=60)

                    if not self._connections_open:
                        return

                self._cleanup_connections()
        except KeyboardInterrupt:
            return
    
    def _cleanup_connections(self, all: bool = False):
        removed_connections: list[PoolConnection] = []
        cleaned_up_connections: list[PoolConnection] = []

        with self._connection_pool_lock:
            try:
                while self._connection_count > 0:
                    if not self._connection_sem.acquire(blocking=all):
                        break

                    current_t = time.time()

                    try:
                        if len(self._connection_pool) <= 0:
                            break

                        pool_item = self._connection_pool.popleft()
                        assert pool_item is not None, "no connection was aquired"

                        if current_t - pool_item.last_used > self._idle_timeout or all:
                            pool_item.connection.close()
                            self._connection_count -= 1
                            removed_connections.append(pool_item)
                        else:
                            cleaned_up_connections.append(pool_item)
                    finally:
                        self._connection_sem.release()
            finally:
                for pool_item in cleaned_up_connections:
                    self._connection_pool.append(pool_item)

                # print('removed_connections', removed_connections)
                # print('cleaned_up_connections', cleaned_up_connections)
