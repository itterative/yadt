import re
import duckdb
import threading
import contextlib

from injector import inject

from yadt.configuration import Configuration

SEARCH_TERM_RE = re.compile('[a-z0-9]+')

class WikiDB:
    @inject
    def __init__(self, configuration: Configuration):
        self.path = configuration.cache_folder / 'wiki.duck.db'

        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._readers = 0

        self._connection = None
        self._setup_connection()

    def _setup_connection(self):
        self._connection = duckdb.connect(str(self.path), read_only=False)
        self._connection.install_extension('fts')
        self._connection.load_extension('fts')

        self._setup_migrations()
        self._do_migrations()

    @contextlib.contextmanager
    def _conn(self, read_only: bool = False):
        # duckdb only allows for one writer

        try:
            if read_only:
                with self._read_lock:
                    if self._readers == 0:
                        self._write_lock.acquire()

                    self._readers += 1
            else:
                self._write_lock.acquire()

            yield self._connection.cursor()
        finally:
            if read_only:
                with self._read_lock:
                    self._readers -= 1

                    if self._readers == 0:
                        self._write_lock.release()
            else:
                self._write_lock.release()

    def _setup_migrations(self):
        with self._conn(read_only=False) as cursor:
            cursor.begin()
            try:
                cursor.execute("""
                    create table if not exists migrations (name text, timestamp timestamp default current_timestamp);
                """)
                cursor.commit()
            except duckdb.Error as e:
                cursor.rollback()
                raise Exception("could not create migrations table") from e
            
    def _do_migration(self, name: str, script: str):
        with self._conn(read_only=False) as cursor:
            cursor.begin()
            try:
                rows = cursor.execute('select * from migrations where name = ?', (name,)).fetchall()
                if len(rows) > 0:
                    return
                
                cursor.execute(script)
                cursor.execute("insert into migrations (name) values (?)", (name,))
                cursor.commit()
            except duckdb.Error as e:
                cursor.rollback()
                raise Exception(f"could not perform migration: {name}") from e
            
    def _do_migrations(self):
        self._do_migration("wiki_table", """
            create table if not exists wiki (
                id integer primary key,
                post_count integer not null,
                title varchar not null,
                markdown varchar,
                search_title varchar,
                search_text varchar
            );
        """)

    def count_pages(self):
        with self._conn(read_only=True) as cursor:
            results = cursor.sql("select count(*) from wiki").fetchall()

            if len(results) == 0:
                return 0
            
            return int(results[0][0])

    def get_markdown_for_title(self, title: str):
        with self._conn(read_only=True) as cursor:
            results = cursor.sql("select markdown from wiki where title = ? limit 1", params=(title,)).fetchall()

            if len(results) == 0:
                return None

            return str(results[0][0])

    def query_wiki(self, search_term: str, limit: int = 10):
        search_term = ' '.join(re.findall(SEARCH_TERM_RE, search_term))

        with self._conn(read_only=True) as cursor:
            cursor.execute("set variable search_term = ?", parameters=(search_term,))

            query = """
                select title, post_count, title_score0 + title_score1 as title_score, text_score, title_score0 + title_score1 + text_score * log(post_count) as total_score from (
                    select 
                        *,
                        greatest(0.1, fts_main_wiki.match_bm25(id, getvariable('search_term'), fields := 'search_title', k := 0.0, b := 0.5, conjunctive := 1)) as title_score0,
                        greatest(0, fts_main_wiki.match_bm25(id, getvariable('search_term'), fields := 'search_title', k := 0.0, b := 0.5)) as title_score1,
                        add(greatest(0, fts_main_wiki.match_bm25(id, getvariable('search_term'), fields := 'search_text', k := 1.2, b := 0.0)), 0.1) as text_score
                    from wiki
                ) sq where title_score0 > 0.1 or title_score1 > 0.1 or text_score > 0.1 order by title_score0 desc, total_score desc limit ?;
            """

            results = cursor.sql(query, params=(limit,)).fetchall()
        
    def query_title(self, search_terms: list[str], limit: int = 10):
        with self._conn(read_only=True) as cursor:
            cursor.execute("set variable search_term = ?", parameters=(search_terms,))

            query = """
                with
                    wiki_title_split as (select title as title_original, post_count, regexp_split_to_array(title, '[^a-z0-9]+') as title from wiki),
                    search_term_score_parts as (select title, post_count, title_original, list_transform(getvariable('search_term'), search_term_part -> list_aggregate(list_transform(title, term -> jaro_winkler_similarity(search_term_part, term)), 'max')) as title_score from wiki_title_split),
                    wiki_title_score as (select title_original, post_count, title_score as title_score_parts, list_aggregate(title_score, 'sum') / length(getvariable('search_term')) as title_score from search_term_score_parts)
                select title_original, post_count, title_score_parts, title_score from wiki_title_score
                    where title_score >= 0.8
                    order by floor(title_score * 100) desc, post_count desc limit ?;
            """

            results = cursor.sql(query, params=(limit,)).fetchall()
            return [[ str(row[0]), int(row[1]) ] for row in results]
        
    def query_wiki_with_title_terms(self, search_term: str, title_terms: list[str], limit: int = 10):
        with self._conn(read_only=True) as cursor:
            cursor.execute("set variable search_term = ?", parameters=(search_term,))
            cursor.execute("set variable title_terms = ?", parameters=(title_terms,))

            query = """
                with
                    wiki_text_search as (select *, add(greatest(0, fts_main_wiki.match_bm25(id, getvariable('search_term'), fields := 'search_text', k := 1.2, b := 0.0)), 0.1) as text_score from wiki),
                    wiki_title_split as (select title as title_original, post_count, text_score, regexp_split_to_array(title, '[^a-z0-9]+') as title from wiki_text_search),
                    search_term_score_parts as (select title, post_count, title_original, text_score, list_transform(getvariable('title_terms'), search_term_part -> list_aggregate(list_transform(title, term -> jaro_winkler_similarity(search_term_part, term)), 'max')) as title_score from wiki_title_split),
                    wiki_title_score as (select title_original, post_count, text_score, title_score as title_score_parts, list_aggregate(title_score, 'sum') / length(getvariable('title_terms')) as title_score from search_term_score_parts)
                select title_original, post_count, text_score, title_score_parts, title_score from wiki_title_score
                    where title_score >= 0.8 and text_score > 0.1
                    order by floor(title_score * 100) desc, post_count desc limit ?;
            """

            results = cursor.sql(query, params=(limit,)).fetchall()
            return [[ str(row[0]), int(row[1]) ] for row in results]

    def reset(self):
        with self._write_lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

            self._setup_connection()
