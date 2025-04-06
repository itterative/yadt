import os
import gradio as gr

import zlib
import pickle
import hashlib
import pathlib
import duckdb
import functools
import huggingface_hub

from injector import inject, singleton

from PIL import Image

from yadt.db_dataset import DatasetDB
from yadt.configuration import Configuration
from yadt.tagger_shared import Predictor

from yadt import tagger_shared
from yadt import process_prediction
from yadt import ui_utils


@singleton
class DatasetPage:
    @inject
    def __init__(self, configuration: Configuration, db: DatasetDB, predictor: Predictor):
        self._configuration = configuration
        self._db = db
        self._predictor = predictor

        self._settings_model_repo_default = tagger_shared.default_repo
        self._settings_general_thresh_default = self._configuration.score_general_threshold
        # self._settings_general_mcut_enabled_default = 'False'
        self._settings_character_thresh_default = self._configuration.score_character_threshold
        # self._settings_character_mcut_enabled_default = 'False'
        self._settings_replace_underscores_default = 'True'
        self._settings_trim_general_tag_dupes_default = 'True'
        self._settings_escape_brackets_default = 'False'
        self._settings_overwrite_current_caption_default = 'False'
        self._settings_prefix_tags_default = ''
        self._settings_keep_tags_default = ''
        self._settings_ban_tags_default = ''
        self._settings_map_tags_default = ''
        self._settings_whitelist_tags_defaults = ''
        self._settings_whitelist_tag_group_defaults = ui_utils.NO_DROPDOWN_SELECTION

        self._settings_defaults = [
            self._settings_model_repo_default,
            self._settings_general_thresh_default,
            # self._settings_general_mcut_enabled_default,
            self._settings_character_thresh_default,
            # self._settings_character_mcut_enabled_default,
            self._settings_replace_underscores_default,
            self._settings_trim_general_tag_dupes_default,
            self._settings_escape_brackets_default,
            self._settings_overwrite_current_caption_default,
            self._settings_prefix_tags_default,
            self._settings_keep_tags_default,
            self._settings_ban_tags_default,
            self._settings_map_tags_default,
            self._settings_whitelist_tags_defaults,
            self._settings_whitelist_tag_group_defaults,
        ]

        try:
            self._tag_groups_parquet = huggingface_hub.hf_hub_download(
                'itterative/danbooru_wikis_full',
                filename='tag_groups.parquet',
                repo_type='dataset',
                revision='5f697c8f1d2e54cbb9977ca4960ac588c6eeb57b',
            )
        except Exception as e:
            raise AssertionError("Failed to download tag_groups.parquet from HuggingFace") from e

    def _temp_folder_gallery_path(self, name: str):        
        cache_folder = self._configuration.cache_folder / 'dataset_gallery'
        cache_folder.mkdir(exist_ok=True)

        return str(cache_folder / f'{name}.jpeg')

    def _save_caption_for_image_path(self, image_path: str, caption: str, overwrite_current_caption: bool = False):
        caption_file_path = image_path[:image_path.rindex('.')] + '.txt'

        if not os.path.exists(caption_file_path) or overwrite_current_caption:
            with open(caption_file_path, 'w') as f:
                f.write(caption)

    def _load_caption_for_image_path(self, image_path: str):
        caption_file_path = image_path[:image_path.rindex('.')] + '.txt'

        if not os.path.exists(caption_file_path):
            return None
        
        with open(caption_file_path, 'r') as f:
            return f.read().strip()

    def _hash_file(self, path: str):
        with open(path, 'rb') as f:
            hash = hashlib.sha256(f.read())
            return hash.digest()

    def _encode_results(self, *args):
        return zlib.compress(pickle.dumps(args))

    def _decode_results(self, data: bytes):
        return pickle.loads(zlib.decompress(data))
    
    def _load_whitelist_tag_groups(self):
        return list(map(lambda row: str(row[0]), duckdb.sql(f"select distinct tag_group from '{self._tag_groups_parquet}'").fetchall()))
    
    @functools.lru_cache()
    def _process_whitelist_tag(self, whitelist_tag_group: str, *whitelist_tags: str, replace_underscores: bool = False, skip: bool = False):
        if skip:
            return None

        if len(whitelist_tags) == 0:
            whitelist_tags = ''
        else:
            whitelist_tags = ', '.join(whitelist_tags)

        tags = list(map(lambda tag: tag.strip(), whitelist_tags.split(',')))

        if whitelist_tag_group != ui_utils.NO_DROPDOWN_SELECTION:
            tags.extend(list(map(lambda row: row[0], duckdb.sql(f"select distinct tag from '{self._tag_groups_parquet}' where tag_group = ?", params=(whitelist_tag_group,)).fetchall())))

        if len(tags) == 0:
            return None

        if replace_underscores:
            tags = list(map(lambda t: t.replace('_', ' '), tags))

        return tags

    def _process_dataset_folder(
            self,
            folder: str,
            model_repo: str,
            general_thresh: float,
            # general_mcut_enabled: bool,
            character_thresh: float,
            # character_mcut_enabled: bool,
            replace_underscores: bool,
            trim_general_tag_dupes: bool,
            escape_brackets: bool,
            overwrite_current_caption: bool,
            prefix_tags: str,
            keep_tags: str,
            ban_tags: str,
            map_tags: str,
            whitelist_tags: str,
            whitelist_tag_group: str,
            progress: gr.Progress,
    ):
        assert len(folder) > 0, "No folder given"
        assert os.path.isdir(folder), "Folder either doesn't exist or is not a folder"

        whitelist_tags = whitelist_tags or ''
        whitelist_tag_group = whitelist_tag_group or ui_utils.NO_DROPDOWN_SELECTION
        skip_whitelist = len(whitelist_tags) == 0 and whitelist_tag_group == ui_utils.NO_DROPDOWN_SELECTION

        self._db.update_recent_datasets(folder)

        files = os.listdir(folder)
        files = list(filter(lambda f: not f.endswith('.txt') and not f.endswith('.npz') and not f.endswith('.json'), files))

        all_count = 0
        all_images = []
        all_rating = dict()
        all_character_res = dict()
        all_general_res = dict()

        for index, file in progress.tqdm(list(enumerate(files)), desc=folder):
            image_path = str(pathlib.Path(folder) / file)

            file_hash = self._hash_file(image_path)
            file_hash_hex = file_hash.hex()

            try:
                image = Image.open(image_path)
            except Exception as e:
                continue

            cache = self._db.get_dataset_cache(file_hash, model_repo)
            if cache is not None:
                rating, general_res, character_res = self._decode_results(cache)
            else:
                self._predictor.load_model(model_repo, device=self._configuration.device)
                rating, general_res, character_res = self._predictor.predict(image)

                self._db.set_dataset_cache(file_hash, model_repo, folder, self._encode_results(rating, general_res, character_res))

            sorted_general_strings, rating, general_res, character_res = \
                process_prediction.post_process_prediction(
                    rating, general_res, character_res,
                    # general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled,
                    general_thresh, False, character_thresh, False,
                    replace_underscores, trim_general_tag_dupes, escape_brackets,
                    prefix_tags, keep_tags, ban_tags, map_tags,
                )
            
            manual_edit = self._db.get_dataset_edit(folder, file_hash)
            if manual_edit is not None:
                previous_edit, new_edit = manual_edit
                
                sorted_general_strings_post = process_prediction.post_process_manual_edits(
                    previous_edit, new_edit, sorted_general_strings,
                    whitelist=self._process_whitelist_tag(whitelist_tag_group, whitelist_tags, 'BREAK', prefix_tags, keep_tags, replace_underscores=replace_underscores, skip=skip_whitelist)
                )
            elif existing_caption := self._load_caption_for_image_path(str(image_path)):
                sorted_general_strings_post = process_prediction.post_process_manual_edits(
                    existing_caption, existing_caption, sorted_general_strings,
                    whitelist=self._process_whitelist_tag(whitelist_tag_group, whitelist_tags, 'BREAK', prefix_tags, keep_tags, replace_underscores=replace_underscores, skip=skip_whitelist)
                )
                
                sorted_general_strings = existing_caption
            else:
                sorted_general_strings_post = process_prediction.post_process_manual_edits(
                    '', '', sorted_general_strings,
                    whitelist=self._process_whitelist_tag(whitelist_tag_group, whitelist_tags, 'BREAK', prefix_tags, keep_tags, replace_underscores=replace_underscores, skip=skip_whitelist)
                )

            all_count += 1

            temp_image_path = self._temp_folder_gallery_path(file_hash_hex)
            if not os.path.exists(temp_image_path):
                image.convert("RGB").save(temp_image_path, quality=75, optimize=True)

            all_images.append((file_hash_hex, [image_path, sorted_general_strings, sorted_general_strings_post]))

            for k in rating.keys():
                all_rating[k] = all_rating.get(k, 0) + rating[k]

            for k in character_res.keys():
                all_character_res[k] = all_character_res.get(k, 0) + 1
            
            for k in general_res.keys():
                all_general_res[k] = all_general_res.get(k, 0) + 1

            self._save_caption_for_image_path(image_path, sorted_general_strings_post, overwrite_current_caption=overwrite_current_caption)

        for k in all_rating.keys():
            all_rating[k] = all_rating[k] / all_count

        for k in all_character_res.keys():
            all_character_res[k] = all_character_res[k] / all_count

        for k in all_general_res.keys():
            all_general_res[k] = all_general_res[k] / all_count

        return all_images, all_rating, all_general_res, all_character_res


    def _load_recent_datasets(self):
        return self._db.get_recent_datasets()

    def _load_dataset_settings(self, folder: str):
        model_repo = str(self._db.get_dataset_setting(folder, 'model_repo', default=self._settings_model_repo_default))
        general_thresh = float(self._db.get_dataset_setting(folder, 'general_thresh', default=self._settings_general_thresh_default))
        # general_mcut_enabled = (self.db.get_dataset_setting(folder, 'general_mcut_enabled', default=self._settings_general_mcut_enabled_default)) == 'True'
        character_thresh = float(self._db.get_dataset_setting(folder, 'character_thresh', default=self._settings_character_thresh_default))
        # character_mcut_enabled = (self.db.get_dataset_setting(folder, 'character_mcut_enabled', default=self._settings_character_mcut_enabled_default)) == 'True'
        replace_underscores = (self._db.get_dataset_setting(folder, 'replace_underscores', default=self._settings_replace_underscores_default)) == 'True'
        trim_general_tag_dupes = (self._db.get_dataset_setting(folder, 'trim_general_tag_dupes', default=self._settings_trim_general_tag_dupes_default)) == 'True'
        escape_brackets = (self._db.get_dataset_setting(folder, 'escape_brackets', default=self._settings_escape_brackets_default)) == ''
        overwrite_current_caption = (self._db.get_dataset_setting(folder, 'overwrite_current_caption', default=self._settings_overwrite_current_caption_default)) == 'True'
        prefix_tags = str(self._db.get_dataset_setting(folder, 'prefix_tags', default=self._settings_prefix_tags_default))
        keep_tags = str(self._db.get_dataset_setting(folder, 'keep_tags', default=self._settings_keep_tags_default))
        ban_tags = str(self._db.get_dataset_setting(folder, 'ban_tags', default=self._settings_ban_tags_default))
        map_tags = str(self._db.get_dataset_setting(folder, 'map_tags', default=self._settings_map_tags_default))
        whitelist_tags = str(self._db.get_dataset_setting(folder, 'whitelist_tags', default=self._settings_whitelist_tags_defaults))
        whitelist_tag_group = str(self._db.get_dataset_setting(folder, 'whitelist_tag_group', default=self._settings_whitelist_tag_group_defaults))

        return [
            model_repo,
            general_thresh,
            # general_mcut_enabled,
            character_thresh,
            # character_mcut_enabled,
            replace_underscores,
            trim_general_tag_dupes,
            escape_brackets,
            overwrite_current_caption,
            prefix_tags,
            keep_tags,
            ban_tags,
            map_tags,
            whitelist_tags,
            whitelist_tag_group,
        ]

    def _save_dataset_settings(
            self,
            folder: str,
            model_repo: str,
            general_thresh: float,
            # general_mcut_enabled: bool,
            character_thresh: float,
            # character_mcut_enabled: bool,
            replace_underscores: bool,
            trim_general_tag_dupes: bool,
            escape_brackets: bool,
            overwrite_current_caption: bool,
            prefix_tags: str,
            keep_tags: str,
            ban_tags: str,
            map_tags: str,
            whitelist_tags: str,
            whitelist_tag_group: str,
    ):
        self._db.set_dataset_setting(folder, 'model_repo', str(model_repo))
        self._db.set_dataset_setting(folder, 'general_thresh', str(general_thresh))
        # self.db.set_dataset_setting(folder, 'general_mcut_enabled', str(general_mcut_enabled))
        self._db.set_dataset_setting(folder, 'character_thresh', str(character_thresh))
        # self.db.set_dataset_setting(folder, 'character_mcut_enabled', str(character_mcut_enabled))
        self._db.set_dataset_setting(folder, 'replace_underscores', str(replace_underscores))
        self._db.set_dataset_setting(folder, 'trim_general_tag_dupes', str(trim_general_tag_dupes))
        self._db.set_dataset_setting(folder, 'escape_brackets', str(escape_brackets))
        self._db.set_dataset_setting(folder, 'overwrite_current_caption', str(overwrite_current_caption))
        self._db.set_dataset_setting(folder, 'prefix_tags', str(prefix_tags))
        self._db.set_dataset_setting(folder, 'keep_tags', str(keep_tags))
        self._db.set_dataset_setting(folder, 'ban_tags', str(ban_tags))
        self._db.set_dataset_setting(folder, 'map_tags', str(map_tags))
        self._db.set_dataset_setting(folder, 'whitelist_tags', str(whitelist_tags))
        self._db.set_dataset_setting(folder, 'whitelist_tag_group', str(whitelist_tag_group))


    def ui(self):
        with gr.Blocks() as page:
            with gr.Row():
                with gr.Column(variant="panel"):
                    with gr.Row(equal_height=True):
                        folder = gr.Dropdown(
                            label="Select folder:",
                            allow_custom_value=True,
                            scale=1,
                        )

                        load_folder = gr.Button(value="Load", variant="primary", scale=0)

                    gr.HTML('<p style="margin-top: -1em"><i>Dataset settings are saved on submit. Use the load button to reload them.</i></p>')

                    model_repo = gr.Dropdown(
                        tagger_shared.dropdown_list,
                        value=tagger_shared.default_repo,
                        label="Model",
                    )

                    with gr.Row():
                        general_thresh = gr.Slider(
                            0,
                            1,
                            step=self._configuration.score_slider_step,
                            value=self._configuration.score_general_threshold,
                            label="General Tags Threshold",
                            scale=3,
                        )

                        character_thresh = gr.Slider(
                            0,
                            1,
                            step=self._configuration.score_slider_step,
                            value=self._configuration.score_character_threshold,
                            label="Character Tags Threshold",
                            scale=3,
                        )

                    with gr.Row():
                        overwrite_current_caption = gr.Checkbox(
                            value=False,
                            label="Overwrite existing captions",
                            scale=1,
                        )
                        replace_underscores = gr.Checkbox(
                            value=True,
                            label="Replace underscores with spaces",
                            scale=1,
                        )
                        trim_general_tag_dupes = gr.Checkbox(
                            value=True,
                            label="Trim duplicate general tags",
                            scale=1,
                        )
                        escape_brackets = gr.Checkbox(
                            value=False,
                            label="Escape brackets (for webui)",
                            scale=1,
                        )

                    with gr.Column(variant='panel'):
                        prefix_tags = gr.Textbox(label="Prefix tags:", placeholder="tag1, tag2, ...")
                        keep_tags = gr.Textbox(label="Keep tags:", placeholder="tag1, tag2, ...")
                        ban_tags = gr.Textbox(label="Ban tags:", placeholder="tag1, tag2, ...")
                        map_tags = gr.Textbox(label="Map tags", placeholder="one or more lines of \"tag1, tag2, ... : tag\"", lines=5, max_lines=100)

                        with gr.Row():
                            whitelist_tags = gr.Textbox(label="Whitelist tags:", value='', placeholder="tag1, tag2, ...")
                            whitelist_tag_groups = gr.Dropdown(label="Whitelist tag groups:", value=ui_utils.NO_DROPDOWN_SELECTION, choices=[ui_utils.NO_DROPDOWN_SELECTION] + self._load_whitelist_tag_groups(), interactive=True)

                        gr.HTML('''
                            <p>Keeping tags</p>
                            <p><i>Adding any tags to this will sort the tags and add them before a "BREAK" tag.</i></p>
                            <br>
                            <p>Mapping tags</p>
                            <p><i>You can map certain one or more tags to different tags. Examples: </i></p>
                            <p style="padding-left: 1em"><i>* BAD_TAG : GOOD_TAG</i></p>
                            <p style="padding-left: 1em"><i>* 2girl : 2girls, GIRL_ONE, GIRL_TWO</i></p>
                            <br>
                            <p>Whitelisting tags</p>
                            <p><i>If you want to add only certain tags or tag groups to your results, you can use this option.</i></p>
                            <p><i>You can check what tags are whitelisted in the tag groups by searching through the wiki.</i></p>
                        ''')
                    
                    with gr.Row():
                        clear = gr.ClearButton(
                            components=[
                                folder,
                                model_repo,
                                general_thresh,
                                # general_mcut_enabled,
                                character_thresh,
                                # character_mcut_enabled,
                                replace_underscores,
                                trim_general_tag_dupes,
                                escape_brackets,
                                overwrite_current_caption,
                                prefix_tags,
                                keep_tags,
                                ban_tags,
                                map_tags,
                            ],
                            variant="secondary",
                            size="lg",
                        )
            
                        submit = gr.Button(value="Submit", variant="primary", size="lg")

                with gr.Column():
                    with gr.Column(variant="panel"):
                        # FIXME: gr.JSON deals stateless requests, but it also sends all the captions over
                        gallery_cache = gr.JSON(visible=False)
                        gallery_selection = gr.JSON(visible=False)

                        with gr.Column(visible=False) as gallery_tags_filter:
                            gr.HTML('<h3>Dataset gallery</h3><p><i>Use the dropdown below to filter the images by tags</i></p>')
                            gallery_tags_filter_dropdown = gr.Dropdown(choices=[], label="Filter by tag", multiselect=True, interactive=True, show_label=False, container=False)

                        gallery = gr.Gallery(interactive=False, columns=3)

                        with gr.Column(visible=False) as gallery_tags_view:
                            gallery_tags_edit = gr.Text(interactive=False, show_label=False, container=False, placeholder="Select an image to view the resulting tags.")

                            with gr.Row():
                                gallery_tags_reset = gr.Button(value="Reset")
                                gallery_tags_reload = gr.Button(value="Reload")
                                gallery_tags_save = gr.Button(value="Save", variant="primary")

                            gr.HTML('''
                                <p>Editing dataset tags is still <i><b>experimental</b></i>.</p>
                                <p style="font-size: 0.9em">
                                    <i>
                                        <b>Reset</b> will clear any changes made previously, and set the tags back to the original model tags (using the rules set on the left side). <br>
                                        <b>Reload</b> will clear any temporary manual changes and load the latest modified tags from the local database. <br>
                                        <b>Save</b> will update the database and the caption file. Neither reset nor reload will update the database or caption file until the save button is clicked.
                                    </i>
                                </p>
                            ''')

                    with gr.Column(variant="panel"):
                        rating = gr.Label(label="Rating")
                        character_res = gr.Label(label="Output (characters)")
                        general_res = gr.Label(label="Output (tags)")

                        clear.add(
                            [
                                rating,
                                character_res,
                                general_res,
                            ]
                        )


        @gr.on(
            (page.load, gallery_cache.change),
            inputs=[gallery_tags_filter_dropdown, gallery_cache],
            outputs=[gallery_tags_filter_dropdown],
        )
        def _process_dataset_gallery(previous_choices: list[tuple[str, str]], all_images: list[tuple[str, tuple[str, str, str]]]):
            with ui_utils.gradio_warning():
                if all_images is None:
                    all_images = []

                all_image_dict = {}

                for _, (_, _, tags) in all_images:
                    for tag in tags.split(','):
                        tag = tag.strip()

                        if tag in all_image_dict:
                            all_image_dict[tag] += 1
                        else:
                            all_image_dict[tag] = 1

                return gr.Dropdown(choices=[
                    (f'{tag} [{count}]', tag) for tag, count in sorted(all_image_dict.items(), key=lambda item: item[1], reverse=True)
                ])

            return gr.Dropdown(choices=previous_choices)

        @gr.on(
            (gallery_cache.change, gallery_tags_filter_dropdown.change),
            inputs=[gallery, gallery_cache, gallery_tags_filter_dropdown],
            outputs=[gallery],
        )
        def _process_dataset_gallery(previous_gallery, all_images: list[tuple[str, tuple[str, str, str]]], filters: list[str]):
            with ui_utils.gradio_warning():
                if all_images is None:
                    all_images = []

                if filters is None or len(filters) == 0:
                    return [
                        (self._temp_folder_gallery_path(image), image) for image, tags in all_images
                    ]
                
                filters = set(filters)

                return [
                    (self._temp_folder_gallery_path(image), image) for image, (_, _, tags) in all_images if set([ tag.strip() for tag in tags.split(',') ]).issuperset(filters)
                ]
            
            return previous_gallery


        @gr.on(
            (gallery_cache.change),
            inputs=[gallery_selection, folder, gallery_cache],
            outputs=[gallery_selection, gallery_tags_edit, gallery_tags_view],
        )
        def _on_gallery_cache_change(selection: tuple[str, str], folder: str, all_images: list[tuple[str, tuple[str, str, str]]]):
            if selection is None or selection[1] is None:
                return _on_gallery_deselect([folder, None])
            
            if next(filter(lambda image: image[0] == selection[1], all_images), None) is None:
                return _on_gallery_deselect(selection)

            event = gr.SelectData(None, data={'index': 0, 'value': {'caption': selection[1]}})
            return _on_gallery_select(selection, all_images, event)

        @gr.on(
            (gallery.select),
            inputs=[gallery_selection, gallery_cache],
            outputs=[gallery_selection, gallery_tags_edit, gallery_tags_view],
        )
        def _on_gallery_select(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]], event: gr.SelectData):
            with ui_utils.gradio_warning():
                _selection = event.value['caption']

                caption = next(filter(lambda image: image[0] == _selection, all_images), [None, [None, None, None]])[1][2]

                assert caption is not None, f"Could not find caption for the selected image: {_selection}"

                return [
                    [selection[0], _selection],
                    gr.Text(value=caption, interactive=True),
                    gr.Column(visible=True),
                ]
            
            return [
                selection,
                gr.Text(value=None, interactive=False),
                gr.Column(visible=False),
            ]

        @gr.on(
            gallery.preview_close,
            inputs=[gallery_selection],
            outputs=[gallery_selection, gallery_tags_edit, gallery_tags_view],
        )
        def _on_gallery_deselect(selection: tuple[str, str]):
            with ui_utils.gradio_warning():
                return [
                    [selection[0], None],
                    gr.Text(value=None, interactive=False),
                    gr.Column(visible=False),
                ]
            
            return [
                selection,
                gr.Text(value=None, interactive=False),
                gr.Column(visible=False),
            ]

        @gr.on(
            gallery_tags_reset.click,
            inputs=[gallery_selection, gallery_cache],
            outputs=[gallery_tags_edit],
        )
        def _on_gallery_reset(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]]):
            with ui_utils.gradio_warning():
                if selection[1] is None:
                    return None

                selection = selection[1]
                return next(filter(lambda image: image[0] == selection, all_images), [None, [None, None, None]])[1][1]
            
            return None

        @gr.on(
            gallery_tags_reload.click,
            inputs=[gallery_selection, gallery_cache],
            outputs=[gallery_tags_edit],
        )
        def _on_gallery_reload(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]]):
            with ui_utils.gradio_warning():
                if selection[1] is None:
                    return None

                selection = selection[1]
                return next(filter(lambda image: image[0] == selection, all_images), [None, [None, None, None]])[1][2]
            
            return None

        @gr.on(
            gallery_tags_save.click,
            inputs=[gallery_selection, gallery_cache, gallery_tags_edit],
            outputs=[gallery_cache],
        )
        def _on_gallery_save(selection: tuple[str, str], all_images: list[tuple[str, tuple[str, str, str]]], caption: str):
            with ui_utils.gradio_warning():
                assert len(selection) > 0, "No gallery image selected"

                folder, selection = selection
                gallery_item = next(filter(lambda image: image[0] == selection, all_images), [None, [None, None, None]])

                assert gallery_item[0] is not None, f"Could not find selected image: {selection}"

                file_hash_hex = gallery_item[0]
                file_hash = bytes.fromhex(file_hash_hex)
                image_path, initial_edit, _ = gallery_item[1]

                self._save_caption_for_image_path(image_path, caption, overwrite_current_caption=True)
                self._db.set_dataset_edit(folder, file_hash, initial_edit, caption)

                try:
                    all_images_i = list(map(lambda i: i[0], all_images)).index(file_hash_hex)
                    all_images[all_images_i] = [file_hash_hex, [image_path, initial_edit, caption]]
                except ValueError:
                    gr.Warning(f'Could not update caption for selected image: {selection}')

                return all_images
            
            return all_images


        @gr.on(
            page.load,
            outputs=[
                folder,
                model_repo,
                general_thresh,
                # general_mcut_enabled,
                character_thresh,
                # character_mcut_enabled,
                replace_underscores,
                trim_general_tag_dupes,
                escape_brackets,
                overwrite_current_caption,
                prefix_tags,
                keep_tags,
                ban_tags,
                map_tags,
                whitelist_tags,
                whitelist_tag_groups,
            ],
        )
        def _load_recent_datasets():
            with ui_utils.gradio_warning():
                choices = self._load_recent_datasets()
                value = choices[0] if len(choices) > 0 else None
                settings = self._load_dataset_settings(value) if value is not None else self._settings_defaults

                return [gr.Dropdown(value=value, choices=choices)] + settings

            return [gr.Dropdown(value=None, choices=[])] + self._settings_defaults

        @gr.on(
            (folder.select, load_folder.click),
            inputs=[folder],
            outputs=[
                model_repo,
                general_thresh,
                # general_mcut_enabled,
                character_thresh,
                # character_mcut_enabled,
                replace_underscores,
                trim_general_tag_dupes,
                escape_brackets,
                overwrite_current_caption,
                prefix_tags,
                keep_tags,
                ban_tags,
                map_tags,
                whitelist_tags,
                whitelist_tag_groups,
            ],
        )
        def _load_dataset_settings(folder: str):
            with ui_utils.gradio_warning():
                return self._load_dataset_settings(folder)
            
            return self._settings_defaults

        @gr.on(
            submit.click,
            inputs=[
                folder,
                model_repo,
                general_thresh,
                # general_mcut_enabled,
                character_thresh,
                # character_mcut_enabled,
                replace_underscores,
                trim_general_tag_dupes,
                escape_brackets,
                overwrite_current_caption,
                prefix_tags,
                keep_tags,
                ban_tags,
                map_tags,
                whitelist_tags,
                whitelist_tag_groups,
            ],
            outputs=[
                folder,
                gallery_cache,
                gallery_tags_filter,
                rating,
                general_res,
                character_res,
            ],
        )
        def _process_dataset_folder(
                folder: str,
                model_repo: str,
                general_thresh: float,
                # general_mcut_enabled: bool,
                character_thresh: float,
                # character_mcut_enabled: bool,
                replace_underscores: bool,
                trim_general_tag_dupes: bool,
                escape_brackets: bool,
                overwrite_current_caption: bool,
                prefix_tags: str,
                keep_tags: str,
                ban_tags: str,
                map_tags: str,
                whitelist_tags: str,
                whitelist_tag_groups: str,
                progress = gr.Progress(),
        ):
            with ui_utils.gradio_warning():
                all_images, all_rating, all_general_res, all_character_res = self._process_dataset_folder(
                    folder,
                    model_repo,
                    general_thresh,
                    # general_mcut_enabled,
                    character_thresh,
                    # character_mcut_enabled,
                    replace_underscores,
                    trim_general_tag_dupes,
                    escape_brackets,
                    overwrite_current_caption,
                    prefix_tags,
                    keep_tags,
                    ban_tags,
                    map_tags,
                    whitelist_tags,
                    whitelist_tag_groups,
                    progress,
                )

                return [
                    gr.Dropdown(choices=self._load_recent_datasets()),
                    all_images,
                    gr.Column(visible=True),
                    all_rating,
                    all_general_res,
                    all_character_res,
                ]
            
            return [ folder, {}, gr.Column(visible=False), {}, {}, {} ]

        @gr.on(
            submit.click,
            inputs=[
                folder,
                model_repo,
                general_thresh,
                # general_mcut_enabled,
                character_thresh,
                # character_mcut_enabled,
                replace_underscores,
                trim_general_tag_dupes,
                escape_brackets,
                overwrite_current_caption,
                prefix_tags,
                keep_tags,
                ban_tags,
                map_tags,
                whitelist_tags,
                whitelist_tag_groups,
            ],
        )
        def _save_dataset_settings(*args):
            with ui_utils.gradio_warning():
                self._save_dataset_settings(*args)
