class _db:
    def __init__(self):
        import pathlib

        self.path = pathlib.Path(__file__).parent / 'dataset.db'

        self._setup_migrations()
        self._do_migrations()

    def _conn(self):
        import sqlite3

        conn = sqlite3.connect(self.path)

        conn.execute('pragma busy_timeout = 10000')
        conn.execute('pragma journal_mode = wal')
        conn.execute('pragma foreign_keys = on')

        return conn

    def _setup_migrations(self):
        import sqlite3

        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                cursor.executescript("""
                    create table if not exists migrations (name text, timestamp integer default current_timestamp);
                """)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception("could not create transactions") from e
        
    def _do_migration(self, name: str, script: str):
        import sqlite3

        with self._conn() as conn:
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
                foreign key (hash_id) references dataset_file_hash(id)
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
    
    def set_dataset_cache(self, hash: bytes, repo_name: str, data: bytes):
        import sqlite3

        with self._conn() as conn:
            cursor = conn.cursor()

            try:
                rows = cursor.execute('insert or ignore into dataset_file_hash (hash) values (?) returning id', (hash,)).fetchall()
                if len(rows) == 0:
                    rows = cursor.execute('select id from dataset_file_hash where hash = ?', (hash,)).fetchall()

                hash_id = int(rows[0][0])
                cursor.execute('insert or replace into dataset_cache (hash_id, repo_name, data) values (?, ?, ?)', (hash_id, repo_name, data))

                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception("failed to update dataset cache") from e

db = _db()
