import re
import duckdb
import typing
import pathlib

import gradio as gr

from injector import inject, singleton

from yadt import ui_utils

from yadt.configuration import Configuration
from yadt.db_wiki import WikiDB
from yadt.process_wiki import process_wiki
from yadt.ui_shared import SharedState


N_RESULTS = 20
SELECTION_REGEX = re.compile('(.+?) \\(.+\\)')
TITLE_TERMS_REGEX = re.compile('".+?"')

@singleton
class WikiPage:
    @inject
    def __init__(self, configuration: Configuration, db: WikiDB, shared_state: SharedState):
        self._configuration = configuration
        self._shared_state = shared_state
        self._db = db

        self._is_building_wiki = False

    def _is_wiki_available(self):
        return self._db.count_pages() > 0

    def _download_and_build_wiki(self):
        assert not self._is_building_wiki, "Wiki database is already being built. Please wait"

        self._is_building_wiki = True
        try:
            yield 0, 'Downloading and building wiki database. This will take around 1-3 minutes.'

            with self._db._conn(read_only=False) as cursor:
                cursor.begin()
                try:
                    for progress, update in process_wiki(cursor):
                        yield progress, update

                    cursor.commit()
                except:
                    cursor.rollback()
                    raise

            yield 1, 'Wiki database has been created!'
        finally:
            self._is_building_wiki = False

    def _query_wiki(self, search: str):
        assert self._db.path.exists(), "Wiki database is not built"
        assert search is not None, "no search term"

        title_terms = []
        for term in re.findall(TITLE_TERMS_REGEX, search):
            title_terms.append(term[1:-1])

        for term in title_terms:
            search = search.replace(f'"{term}"', '')
        search = search.strip()

        if len(title_terms) > 0:
            if len(search) > 0:
                return self._db.query_wiki_with_title_terms(search, title_terms, limit=N_RESULTS)

            return self._db.query_title(title_terms, limit=N_RESULTS)

        return self._db.query_wiki(search, limit=N_RESULTS)

    def _select_wiki(self, selection: str):
        assert selection is not None, "no selection"
        selection_match = SELECTION_REGEX.match(selection)

        assert selection_match is not None, f"bad format for selection: {selection}"

        return selection

    def _load_wiki(self, selection: str):
        assert self._db.path.exists(), "Wiki database is not built"

        markdown = self._db.get_markdown_for_title(selection)

        if markdown is None:
            markdown = ''

        markdown = markdown.strip()
        return markdown

    def _load_wiki_reference(self, selection: list[str]):
        assert selection is not None, "no selection"
        assert len(selection) > 0, "empty selection"

        selection: str = selection[0]
        selection_match = SELECTION_REGEX.match(selection)

        assert selection_match is not None, f"bad format for selection: {selection}"

        selection = selection_match.group(1)
        return selection


    def ui(self):
        result_items: list[gr.HTML] = []

        with gr.Blocks() as page:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Column(scale=0):
                        search_box = gr.Textbox(label="Search tag", placeholder="Type in a booru tag to search through the wiki")

                        with gr.Column(variant="panel", scale=0) as wiki_info_section:
                            gr.HTML('''
                                <p><i>You can search for different terms, both in the wiki title and as well as the wiki page text.</i></p>
                                <p><i>If you want a to search for specific terms in the tag (or wiki title), you can use quotes.</i></p>
                                <p><i>Examples:</i></p>
                                <p style="padding-left: 1em"><i>* "frieren"</i></p>
                                <p style="padding-left: 1em"><i>* frieren "meme"</i></p>
                                <p style="padding-left: 1em"><i>* monster "girl"</i></p>
                            ''')

                        results_title = gr.HTML('<h3>Results</h3>')

                    with gr.Column(variant="panel", scale=0) as results_container:
                        results_selection = gr.JSON(value=[], visible=False)
                        results = gr.JSON(value=[], visible=False)

                        for _ in range(N_RESULTS):
                            result_items.append(gr.HTML(value='', visible=False, container=False, padding=False, elem_classes='wiki_results_item'))

                    with gr.Column(variant="panel", scale=0) as wiki_load_section:
                        gr.HTML('''
                            <p>In order for the wiki to work, it needs to grab several files from Huggingface and build the local wiki database.</p>
                            <p><i>At the moment, the local wiki database is missing. Please click on the button below in order to start downloading and building the local wiki copy.</i></p>
                            <p><i>Building the wiki database takes 1-3 minutes and consumes up to 1GB of RAM.</i></p>
                        ''')

                        build_wiki_button = gr.Button('Download & build', interactive=not self._is_building_wiki)

                    # filling empty space
                    with gr.Column(scale=1):
                        pass

                with gr.Column(scale=2):
                    with gr.Column(variant="panel", elem_classes='wiki_markdown_container', scale=1):
                        with gr.Column(scale=0):
                            wiki_title = gr.HTML('<h3><i>No tag selected</i></h3>')
                            markdown = gr.Markdown('*Select a tag from the results on the left in order to update this section.*', elem_classes='wiki_markdown')
                        
                        with gr.Column(scale=0):
                            wiki_reference = gr.HTML('')

                        # filling empty space
                        with gr.Column(scale=1):
                            pass

                    # filling empty space
                    with gr.Column(scale=0):
                        pass

        @gr.on(
            page.load,
            outputs=[results]
        )
        def _load_some_results():
            if self._is_wiki_available():
                return _query_wiki('')
            return []

        @gr.on(
            self._shared_state.cache_cleared.change
        )
        def _reset_wiki_db():
            self._db.reset()

        @gr.on(
            (page.load, self._shared_state.cache_cleared.change),
            outputs=[wiki_load_section],
        )
        def _check_if_wiki_is_available():
            return gr.update(visible=not self._is_wiki_available())

        @gr.on(
            build_wiki_button.click,
            outputs=[build_wiki_button],
        )
        def _hide_build_wiki_button():
            return gr.update(interactive=False)

        @gr.on(
            build_wiki_button.click,
            outputs=[wiki_load_section, results],
        )
        def _build_wiki():
            with ui_utils.gradio_warning():
                for progress, update in self._download_and_build_wiki():
                    gr.Info(update, duration=2)

                return [gr.update(visible=False)] + [_load_some_results()]

            return [gr.update(visible=True)] + [[]]

        @gr.on(
            search_box.submit,
            inputs=[search_box],
            outputs=[results],
        )
        def _query_wiki(search: str):
            with ui_utils.gradio_warning():
                return [f'{title} ({post_count})' for title, post_count in self._query_wiki(search)]
            return []


        @gr.on(
            (page.load, results.change),
            inputs=[results],
            outputs=[results_title, results_container] + result_items,
        )
        def _load_results(results: list[str]):
            components = [
                gr.update(visible=len(results) > 0),
                gr.update(visible=len(results) > 0),
            ]

            for i in range(N_RESULTS):
                if len(results) - i <= 0:
                    components.append(gr.update(value='', visible=False))
                else:
                    components.append(gr.update(value=results[i], visible=True))

            return components

        @gr.on(
            results_selection.change,
            inputs=[wiki_title, markdown, results_selection],
            outputs=[wiki_title, markdown],    
        )
        def _load_wiki(previous_title: str, previous_markdown: str, selection: list[str]):
            with ui_utils.gradio_warning():
                selection = self._load_wiki_reference(selection)
                markdown = self._load_wiki(selection)

                if len(markdown) == 0:
                    markdown = '*Either the tag has no wiki page or the page is empty.*'

                return [f'<h3>{selection}</h3>', markdown]
            
            return [ previous_title, previous_markdown ]

        @gr.on(
            results_selection.change,
            inputs=[results_selection],
            outputs=[wiki_reference],
        )
        def _load_wiki_reference(selection: list[str]):
            with ui_utils.gradio_warning():
                selection = self._load_wiki_reference(selection)
                return f'<p style="font-size: 0.9em"><i><a href="https://danbooru.donmai.us/wiki_pages/{selection.lower().replace(" ", "_")}">Danbooru wiki</a></i></p>'

            return ''

        for result in result_items:
            @gr.on(
                result.click,
                inputs=[result],
                outputs=[results_selection],
            )
            def _select_wiki(selection: str):
                with ui_utils.gradio_warning():
                    return [self._select_wiki(selection)]
                return [None]
