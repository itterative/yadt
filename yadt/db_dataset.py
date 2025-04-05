import os
import sqlite3
import contextlib
import threading
import pathlib

from injector import inject

from yadt.configuration import Configuration
from yadt.db_pool import Sqlite3DBPool

class DatasetDB:
    @inject
    def __init__(self, configuration: Configuration):
        self.path = configuration.cache_folder / 'dataset.db'
        self._db_lock = threading.Lock()

        self._pool = Sqlite3DBPool(self.path, busy_timeout='10000', journal_model='wal', foreign_keys='on')
        self._pool.open()

        # migrate old path
        old_path = pathlib.Path(__file__).parent.parent / 'dataset.db'
        if old_path.exists():
            old_path.rename(self.path)

        with self._db_lock:
            self._setup_migrations()
            self._do_migrations()

    def _conn(self, locked: bool = True):
        lock = self._db_lock
        if not locked:
            lock = contextlib.nullcontext()

        with lock:
            return self._pool.connection()

    def _setup_migrations(self):
        with self._conn(locked=False) as conn:
            cursor = conn.cursor()

            try:
                cursor.executescript("""
                    create table if not exists migrations (name text, timestamp integer default current_timestamp);
                """)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception("could not create migrations table") from e
        
    def _do_migration(self, name: str, script: str):
        with self._conn(locked=False) as conn:
            cursor = conn.cursor()

            try:
                rows = cursor.execute('select * from migrations where name = ?', (name,)).fetchall()
                if len(rows) > 0:
                    return
                
                cursor.executescript(script)
                cursor.execute("insert into migrations (name) values (?)", (name,))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception(f"could not perform migration: {name}") from e

    def _do_migrations(self):
        self._do_migration("dataset_cache", """
            create table if not exists dataset_file_hash (
                id integer primary key,
                hash blob not null
            );

            create unique index if not exists idx_dataset_file_hash_hash on dataset_file_hash(hash);
                           
            create table if not exists dataset_cache (
                id integer primary key,
                hash_id integer not null,
                repo_name text not null,
                data blob not null,
                foreign key (hash_id) references dataset_file_hash(id) on delete cascade
            );

            create unique index if not exists idx_dataset_cache_repo_name on dataset_cache (hash_id, repo_name);
        """)

        self._do_migration("dataset_settings", """
            create table if not exists dataset_settings (
                id integer primary key,
                dataset text not null,
                key text not null,
                value text not null
            );
               
            create unique index idx_dataset_settings on dataset_settings (dataset, key);
        """)

        self._do_migration("dataset_cache_stats", """
            create table if not exists dataset_stats (
                id integer primary key,
                dataset text not null
            );

            create unique index idx_dataset_stats_dataset on dataset_stats (dataset);

            create table if not exists dataset_cache_stats (
                dataset_id integer not null,
                hash_id integer not null,
                foreign key (dataset_id) references dataset_stats (id) on delete cascade,
                foreign key (hash_id) references dataset_cache (id) on delete cascade
            );

            create unique index idx_dataset_cache_stats_id on dataset_cache_stats (dataset_id, hash_id);
        """)

        self._do_migration("dataset_history", """
            create table if not exists dataset_history (
                id integer primary key,
                dataset_id integer not null,
                foreign key (dataset_id) references dataset_stats (id) on delete cascade
            );
                           
            create index idx_dataset_history_dataset_id on dataset_history (dataset_id);
        """)

        self._do_migration("dataset_cache_v2", """
            drop index if exists idx_dataset_cache_repo_name;
            create unique index if not exists idx_dataset_cache_repo_name on dataset_cache (repo_name, hash_id);
        """)

        self._do_migration("dataset_manual_editing", """
            create table if not exists dataset_manual_edit (
                id integer primary key,
                dataset_id integer not null,
                hash_id integer not null,
                previous_edit text not null,
                new_edit text not null,
                foreign key (dataset_id) references dataset_stats (id) on delete cascade,
                foreign key (hash_id) references dataset_file_hash (id) on delete cascade
            );

            create unique index idx_dataset_manual_edit_id on dataset_manual_edit (dataset_id, hash_id);
        """)

    def get_recent_datasets(self) -> list[str]:
        with self._conn() as conn:
            cursor = conn.cursor()

            rows = cursor.execute('select max(s.dataset) from dataset_stats s left join dataset_history h on s.id = h.dataset_id group by s.id order by max(h.id) desc, s.id desc limit 10').fetchall()
            return [row[0] for row in rows]

    def update_recent_datasets(self, last_dataset: str):
        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                try:
                    rows = cursor.execute('insert or abort into dataset_stats (dataset) values (?) returning id', (last_dataset,)).fetchall()
                    conn.commit()
                except sqlite3.IntegrityError:
                    rows = cursor.execute('select id from dataset_stats where dataset = ?', (last_dataset,)).fetchall()

                dataset_id = int(rows[0][0])

                cursor.execute('insert into dataset_history (dataset_id) values (?)', (dataset_id,))
                cursor.execute('delete from dataset_history where id not in (select id from dataset_history order by id desc limit 10)')

                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception("failed to update dataset cache") from e

    def get_dataset_setting(self, dataset: str, key: str, default=None):
        with self._conn() as conn:
            cursor = conn.cursor()

            rows = cursor.execute('select value from dataset_settings where dataset = ? and key = ?', (dataset, key)).fetchall()
            if len(rows) == 0:
                return default
            
            return str(rows[0][0])
        
    def set_dataset_setting(self, dataset: str, key: str, value: str):
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('insert or replace into dataset_settings (dataset, key, value) values (?, ?, ?)', (dataset, key, value))


    def get_dataset_cache(self, hash: bytes, repo_name: str):
        with self._conn() as conn:
            cursor = conn.cursor()

            rows = cursor.execute('select c.data from dataset_cache c inner join dataset_file_hash h on h.id = c.hash_id where h.hash = ? and c.repo_name = ? limit 1', (hash, repo_name)).fetchall()
            if len(rows) == 0:
                return None
            
            return bytes(rows[0][0])

    def get_dataset_cache_for_repo_name(self):
        with self._conn() as conn:
            cursor = conn.cursor()
            rows = cursor.execute('select repo_name from dataset_cache group by repo_name').fetchall()

            return [
                row[0] for row in rows
            ]

    def get_dataset_cache_usage_for_repo_name(self):
        with self._conn() as conn:
            cursor = conn.cursor()
            rows = cursor.execute('select sum(length(data)), repo_name from dataset_cache group by repo_name').fetchall()

            return [
                { 'repo_name': row[1], 'bytes': row[0] } for row in rows
            ]
    
    def delete_dataset_cache_by_repo_name(self, repo_name: str):
        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute('delete from dataset_cache where repo_name = ?', (repo_name,))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception(f"failed to deleted cache for repo_name: {repo_name}") from e

    def get_dataset_cache_for_dataset(self):
        with self._conn() as conn:
            cursor = conn.cursor()
            
            rows = cursor.execute('select d.dataset from dataset_cache c left join dataset_cache_stats s on c.id = s.hash_id left join dataset_stats d on d.id = s.dataset_id group by d.dataset').fetchall()

            return [
                row[0] for row in rows
            ]

    def get_dataset_cache_usage_for_dataset(self):
        with self._conn() as conn:
            cursor = conn.cursor()
            
            rows = cursor.execute('select sum(length(c.data)), d.dataset from dataset_cache c left join dataset_cache_stats s on c.id = s.hash_id left join dataset_stats d on d.id = s.dataset_id group by d.dataset').fetchall()

            return [
                { 'dataset': row[1], 'bytes': row[0] } for row in rows
            ]
    
    def delete_dataset_cache_by_dataset(self, dataset: str):
        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute('delete from dataset_cache where id in (select s.hash_id from dataset_cache_stats s left join dataset_stats d on d.id = s.dataset_id group by s.hash_id having s.hash_id in (select s2.hash_id from dataset_cache_stats s2 left join dataset_stats d2 on d2.id = s2.dataset_id where d2.dataset = ?) and count(distinct s.dataset_id) = 1)', (dataset,))
                cursor.execute('delete from dataset_stats where dataset = ?', (dataset,))

                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception(f"failed to deleted cache for dataset: {dataset}") from e

    def set_dataset_cache(self, hash: bytes, repo_name: str, dataset: str, data: bytes):
        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                try:
                    rows = cursor.execute('insert or abort into dataset_stats (dataset) values (?) returning id', (dataset,)).fetchall()
                    conn.commit()
                except sqlite3.IntegrityError:
                    rows = cursor.execute('select id from dataset_stats where dataset = ?', (dataset,)).fetchall()

                dataset_id = int(rows[0][0])

                try:
                    rows = cursor.execute('insert or abort into dataset_file_hash (hash) values (?) returning id', (hash,)).fetchall()
                    conn.commit()
                except sqlite3.IntegrityError:
                    rows = cursor.execute('select id from dataset_file_hash where hash = ?', (hash,)).fetchall()

                hash_id = int(rows[0][0])

                try:
                    cursor.execute('insert or abort into dataset_cache (hash_id, repo_name, data) values (?, ?, ?)', (hash_id, repo_name, data))
                    conn.commit()
                except sqlite3.IntegrityError:
                    pass

                try:
                    cursor.execute('insert or abort into dataset_cache_stats (dataset_id, hash_id) values (?, ?)', (dataset_id, hash_id))
                    conn.commit()
                except sqlite3.IntegrityError:
                    pass

                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception("failed to update dataset cache") from e
    
    def get_dataset_edit(self, dataset: str, hash: bytes):
        with self._conn() as conn:
            cursor = conn.cursor()

            rows = cursor.execute('select e.previous_edit, e.new_edit from dataset_manual_edit e left join dataset_stats s on s.id = e.dataset_id left join dataset_file_hash h on h.id = e.hash_id where s.dataset = ? and h.hash = ? limit 1', (dataset, hash)).fetchall()
            if len(rows) == 0:
                return None

            return str(rows[0][0]), str(rows[0][1])

    def set_dataset_edit(self, dataset: str, hash: bytes, previous_edit: str, new_edit: str):
        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                try:
                    rows = cursor.execute('insert or abort into dataset_stats (dataset) values (?) returning id', (dataset,)).fetchall()
                    conn.commit()
                except sqlite3.IntegrityError:
                    rows = cursor.execute('select id from dataset_stats where dataset = ?', (dataset,)).fetchall()

                dataset_id = int(rows[0][0])

                try:
                    rows = cursor.execute('insert or abort into dataset_file_hash (hash) values (?) returning id', (hash,)).fetchall()
                    conn.commit()
                except sqlite3.IntegrityError:
                    rows = cursor.execute('select id from dataset_file_hash where hash = ?', (hash,)).fetchall()

                hash_id = int(rows[0][0])

                cursor.execute('insert or replace into dataset_manual_edit (dataset_id, hash_id, previous_edit, new_edit) values (?, ?, ?, ?)', (dataset_id, hash_id, previous_edit, new_edit))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception("failed to update dataset cache") from e

    def vacuum(self):
        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute('vacuum')
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception('failed to minimize db size') from e
            
    def reset(self):
        with self._db_lock:
            self._pool.close()
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self._pool.open()

            self._setup_migrations()
            self._do_migrations()

