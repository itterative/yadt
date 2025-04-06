import tqdm
import typing
import traceback

import re
import pathlib
import duckdb
import dataclasses
import huggingface_hub


def process_wiki(connection: duckdb.DuckDBPyConnection):
    def apply_regex(pattern: str, **kwargs):
        regex = re.compile(pattern)

        def _apply_regex(fn):
            def _wrapper(text: str) -> str:
                match = regex.match(text)

                if match is None:
                    return text
                
                return fn(*match.groups(), **kwargs)

            return _wrapper

        return _apply_regex

    def apply_regex_sub(pattern: str, re_flags = 0, **kwargs):
        regex = re.compile(pattern)

        def _apply_regex(fn):
            def _wrapper(text: str) -> str:
                return re.sub(regex, lambda m: fn(*m.groups(), **kwargs), text, flags=re_flags)

            return _wrapper

        return _apply_regex


    @dataclasses.dataclass
    class Node:
        type: typing.Literal['tag']|typing.Literal['string']
        tag: str = ''
        content: str = ''
        start_pos: int = -1
        end_pos: int = -1
        children: list['Node'] = dataclasses.field(default_factory=list)
        parent: 'Node' = None
        attr: dict[str, str] = dataclasses.field(default_factory=dict)

    def apply_tag_tree(separators: tuple[str, str], end_char: str = '/'):
        dtext_tags = [
            'b', 'i', 'u', 's', 'tn', 'spoilers', 'nodtext', 'code',
            'quote',
            # 'head', 'body',
            'table', 'colgroup'
        ]

        dtext_tag_one_line = [
            'b', 'i', 'u', 's'
        ]

        dtext_tags_with_attr = [
            'thead', 'tbody', 'tr', 'col', 'th', 'td',
            'expand',
        ]

        dtext_tags_with_no_parsing = [
            'nodtext', 'code',
        ]

        tags = dtext_tags + dtext_tags_with_attr

        re_separators_first = separators[0] if separators[0] != '[' else f'\\{separators[0]}'
        re_separators_second = separators[1] if separators[1] != ']' else f'\\{separators[1]}'
        dtext_split_re = re.compile('(' + '|'.join(map(lambda tag: f'{re_separators_first}{tag}{re_separators_second}|{re_separators_first}/{tag}{re_separators_second}', tags)) + ')')

        start_separator, end_separator = separators
        start_tags = [ f'{start_separator}{tag}{end_separator}' for tag in tags ]
        end_tags = [ f'{start_separator}{end_char}{tag}{end_separator}' for tag in tags ]

        def _start_tag(buf: str, lookahead: str):
            if len(buf) == 0 or buf[-1] != separators[1]:
                return None
            
            # fix: special case where there are links that look like tags
            if lookahead == '(' or lookahead == ']':
                return None

            try:
                sbuf = buf.rindex(separators[0])
                ebuf = buf.rindex(separators[1])
            except ValueError:
                return None
            
            tbuf = buf[sbuf+1:ebuf]

            for tag in dtext_tags:
                if tbuf == tag:
                    return Node(type='tag', content=tbuf, tag=tag, start_pos=sbuf, end_pos=ebuf)
            
            for tag in dtext_tags_with_attr:
                if tbuf == tag or tbuf.startswith(tag + ' ') or tbuf.startswith(tag + '='):
                    node = Node(type='tag', content=tbuf, tag=tag, start_pos=sbuf, end_pos=ebuf)
                    node.attr = _parse_attr(node.content[len(tag):])
                    return node

            return None
        
        def _end_tag(buf: str):
            if len(buf) == 0 or buf[-1] != separators[1]:
                return None

            buf = buf.lower()
            for i, tag in enumerate(end_tags):
                if buf[-len(tag):] == tag:
                    return tags[i]
            return None
        
        def _strip_attr(val: str):
            val = val.strip()
            if val[:1] == '"':
                val = val[1:]
            if val[-1:] == '"':
                val = val[:-1]
            return val

        def _parse_attr(data: str):
            attr = {}
            parts = data.split('=')

            if len(parts) == 1:
                return attr

            i = 0
            while len(parts) > 2:
                i += 1
                assert i < 100, f"attr infinite loop: {data}"

                if len(parts[-1].strip()) == 0:
                    parts = parts[:-1]
                    continue

                second = _strip_attr(parts[-1])
                first_parts = parts[-2].split(' ')
                first = _strip_attr(first_parts[-1])
                parts[-2] = ' '.join(first_parts[:-1])
                parts = parts[:-1]
                attr[first] = second

            assert len(parts) == 2, f"invalid parsed attr: {data}"
            attr[_strip_attr(parts[0])] = _strip_attr(parts[1])
            return attr
        
        def _closest_started_tag(node: Node, tag: str):
            if node is None:
                return None
            if node.tag == tag:
                return node
            return _closest_started_tag(node.parent, tag)
        
        def _string_node(buf: str, i: int, tag: str, parent: Node):
            return Node(type='string', content=buf, start_pos=i-len(buf), end_pos=i, parent=parent)

        def _apply_tag_tree(fn):
            def _wrapper(text: str, **kwargs):
                root = Node(type='tag', start_pos=0, children=[])
                pos = root
                string_buf = ''

                i = -1 # start at -1 to make it point to last character in the buffer
                for char in re.split(dtext_split_re, text):
                    string_buf = string_buf + char
                    i += len(char)

                    if char == '\n':
                        # fix: some tags aren't closed
                        while pos.tag in dtext_tag_one_line:
                            pos.end_pos = i
                            pos = pos.parent

                        continue

                    tag = _start_tag(string_buf, text[i+1:i+2])
                    if tag is not None:
                        should_skip = any(map(lambda _tag: pos.tag == _tag, dtext_tags_with_no_parsing))
                        if should_skip:
                            continue

                        string_buf = string_buf[:-len(tag.content)-2]
                        if len(string_buf) > 0:
                            pos.children.append(_string_node(string_buf, i, tag, pos))
                        string_buf = ''

                        tag.start_pos = i - (tag.end_pos - tag.start_pos)
                        tag.end_pos = -1
                        tag.parent = pos

                        pos.children.append(tag)
                        pos = tag

                        continue

                    tag = _end_tag(string_buf)
                    if tag is not None:
                        should_skip = any(map(lambda _tag: pos.tag == _tag and tag != _tag, dtext_tags_with_no_parsing))
                        if should_skip:
                            continue

                        _pos = _closest_started_tag(pos, tag)
                        if _pos is None:
                            continue

                        string_buf = string_buf[:-len(tag)-3]
                        if len(string_buf) > 0:
                            pos.children.append(_string_node(string_buf, i, tag, pos))
                        string_buf = ''

                        while pos is not _pos:
                            pos.end_pos = i
                            pos = pos.parent

                        pos = pos.parent

                        assert pos is not None, "error: parent is empty"

                        continue

                    pass

                if len(string_buf) > 0:
                    pos.children.append(_string_node(string_buf, i, '', pos))

                while pos is not None:
                    pos.end_pos = i
                    pos = pos.parent

                return fn(root, **kwargs)

            return _wrapper

        return _apply_tag_tree

    def format_tag_tree(node: Node, raw_text: bool = False):
        def _format_dtext(node: Node):
            if node.type == 'string':
                return node.content

            match node.tag:
                case 'code':
                    return _format_code(node)
                case 'table':
                    return _format_table(node)
                case 'spoilers':
                    return _format_spoilers(node)
                case 'expand':
                    return _format_expand(node)
                case _:
                    return _format_dtext_simple(node)

        def _format_dtext_simple(node: Node):
            if node.type == 'string':
                return node.content

            match node.tag:
                case 'nodtext':
                    return _format_nodtext(node)
                case 'quote':
                    return _format_quote(node)
                case 'code':
                    return _format_code(node)
                case 'expand':
                    return _format_expand(node)
                case 'spoilers':
                    return _format_expand(node) # remove spoiler format
                case 'b':
                    return _format_dtext_bold(node)
                case 'i':
                    return _format_dtext_italic(node)
                case 'u':
                    return _format_dtext_underline(node)
                case 's':
                    return _format_dtext_strikethrough(node)
                case 'tn':
                    return _format_dtext_note(node)
                case _:
                    raise AssertionError(f'unsupported simple dtext: {node.tag}')


        def _format_table(node: Node):
            if node.type == 'string':
                return '' # ignore whatever strings
            
            match node.tag:
                case 'table':
                    return '<table>' + '\n'.join(map(_format_table, node.children)) + '</table>'
                case 'thead':
                    return '\n'.join(map(_format_table, node.children))
                case 'head':
                    return '\n'.join(map(_format_table, node.children))
                case 'tbody':
                    return '\n'.join(map(_format_table, node.children))
                case 'body':
                    return '\n'.join(map(_format_table, node.children))
                case 'tr':
                    return '<tr>\n' + '\n'.join(map(_format_table, node.children)) + '\n</tr>'
                case 'th':
                    return '<th>\n' + ''.join(map(_format_dtext_simple, node.children)) + '\n</th>'
                case 'td':
                    return '<td>\n' + ''.join(map(_format_dtext_simple, node.children)) + '\n</td>'
                case _:
                    raise AssertionError(f'unsupported table dtext: {node.tag}')


        def _format_nodtext(node: Node):
            assert len(node.children) <= 1, f"expected one or less children for nodtext"
            assert all(map(lambda n: n.type == 'string', node.children)), f"expected childred to be string for nodtext; got: {', '.join(map(lambda n: n.type if n.type == 'string' else n.type + ':' + n.tag, node.children))}"

            return ''.join(map(lambda n: n.content, node.children))
        
        def _format_code(node: Node):
            assert len(node.children) <= 1, f"expected one or less children for code"
            assert all(map(lambda n: n.type == 'string', node.children)), f"expected childred to be string for code; got: {', '.join(map(lambda n: n.type if n.type == 'string' else n.type + ':' + n.tag, node.children))}"

            text = ''.join(map(lambda n: n.content, node.children))
            return f'```\n{text}```'


        # no markdown equivalent
        def _format_spoilers(node: Node):
            return ''.join(map(_format_dtext, node.children))
        
        # no markdown equivalent
        def _format_expand(node: Node):
            text = ''.join(map(_format_dtext, node.children))
            title = node.attr.get('expand', None)
            if title is not None:
                text = f'##### {title}\n------\n{text}'
            return text


        def _format_quote(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join(map(lambda t: f'>{t.strip()}', text.splitlines()))


        def _format_dtext_bold(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join(map(lambda t: f'**{t.strip()}**', text.splitlines()))
        
        def _format_dtext_italic(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join(map(lambda t: f'*{t.strip()}*', text.splitlines()))
        
        def _format_dtext_underline(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join(map(lambda t: f'<ins>{t.strip()}</ins>', text.splitlines()))
        
        def _format_dtext_strikethrough(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join(map(lambda t: f'~~{t.strip()}~~', text.splitlines()))
        
        def _format_dtext_note(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join(map(lambda t: f'<sub><sup>{t.strip()}</sub></sup>', text.splitlines()))

        def _format_string_node_only(node: Node):
            if node.type == 'string':
                return node.content

            return ''.join(map(_format_string_node_only, node.children))
        

        if raw_text:
            return _format_string_node_only(node)

        return ''.join(map(_format_dtext, node.children))
    

    @apply_regex_sub('h([0-9]).(.+)')
    def parse_heading(heading_type: str, rest: str):
        return '#' * int(heading_type) + ' ' + rest.strip()

    @apply_regex_sub('([*]+)\\w*(.+)')
    def parse_list_items(quotes: str, text: str):
        return f'{"  " * (len(quotes)-1)}* {text}'


    @apply_regex_sub('([a-zA-Z0-9]*)\\[\\[(.+?)\\]\\]([a-zA-Z0-9]*)', links_qualifier = re.compile('\\(.+\\)'))
    def parse_links(left: str, contents: str, right: str, links_qualifier: re.Pattern = None):
        parts = contents.split('|', 1)

        # NOTE: not sure where the anchor goes
        anchor = ''
        anchors_p0 = parts[0].split('#', 1)
        anchors_p1 = parts[1].split('#', 1) if len(parts) > 1 else []
        if len(anchors_p0) == 2 and anchors_p0[1][:1].isupper():
            anchor = anchors_p0[1]
            parts[0] = anchors_p0[0]
        elif len(anchors_p1) == 2  and anchors_p1[1][:1].isupper():
            anchor = anchors_p1[1]
            parts[1] = anchors_p1[0]
        if len(anchor) > 0:
            anchor = '#dtext-' + anchor.lower()

        if len(parts) == 1:
            contents = contents.strip()
            return f'<a href="https://danbooru.donmai.us/wiki_pages/{contents.lower().replace(" ", "_")}{anchor}">{left}{contents}{right}</a>'
        elif parts[1] == '':
            no_qualifier = re.sub(links_qualifier, '', parts[0]).strip()
            return f'<a href="https://danbooru.donmai.us/wiki_pages/{parts[0].strip().lower().replace(" ", "_")}{anchor}">{left}{no_qualifier}{right}</a>'
        else:
            return f'<a href="https://danbooru.donmai.us/wiki_pages/{parts[0].strip().lower().replace(" ", "_")}{anchor}">{left}{parts[1].strip()}{right}</a>'

    @apply_regex_sub('"(.+?)":\\[(.+?)\\]')
    def parse_links_alt(title: str, url: str):
        return f'<a href="{url}">{title}</a>'

    @apply_regex_sub('"(.+?)":([^ ]+)')
    def parse_links_alt2(title: str, url: str):
        return f'<a href="{url}">{title}</a>'

    @apply_regex_sub('\\[(.+?)\\]\\((?!http)([^\\)]+)\\)')
    def parse_links_alt_md_reversed(url: str, title: str):
        return f'<a href="{url}">{title}</a>'

    @apply_regex_sub('<a href="(.+?)">(.+?)</a>')
    def parse_links_alt_html(url: str, title: str):
        return f'<a href="{url}">{title}</a>'

    @apply_regex_sub('<(http.+?)>')
    def parse_links_alt_htmlish(url: str):
        return f'<a href="{url}">{url}</a>'


    @apply_regex_sub('\\[br\\]')
    def parse_breaks():
        return '\n'

    @apply_regex_sub('\\[hr\\]')
    def parse_breakslines():
        return '-------'


    @apply_tag_tree(('[', ']'))
    def parse_dtext_bbcode_style(tree: Node, raw_text: bool = False):
        return format_tag_tree(tree, raw_text=raw_text)

    @apply_tag_tree(('<', '>'))
    def parse_dtext_html_style(tree: Node, raw_text: bool = False):
        return format_tag_tree(tree, raw_text=raw_text)

    def parse_dtext(dtext: str, raw_text: bool = False) -> str:
        if dtext is None:
            return ''

        dtext = dtext.replace('\r\n', '\n')

        dtext = parse_breaks(dtext)
        dtext = parse_breakslines(dtext)

        dtext = parse_heading(dtext)
        dtext = parse_list_items(dtext)
        
        dtext = parse_links(dtext)
        dtext = parse_links_alt(dtext)
        dtext = parse_links_alt2(dtext)
        dtext = parse_links_alt_md_reversed(dtext)
        # dtext = parse_links_alt_html(dtext)
        dtext = parse_links_alt_htmlish(dtext)

        dtext = parse_dtext_html_style(dtext, raw_text=raw_text)
        dtext = parse_dtext_bbcode_style(dtext, raw_text=raw_text)

        return dtext

    def dtext_to_markdown(dtext: str) -> str:
        return parse_dtext(dtext)

    def dtext_to_raw(dtext: str) -> str:
        return parse_dtext(dtext, raw_text=True)


    try:
        yield 0, 'Downloading wiki data'

        tags_csv = huggingface_hub.hf_hub_download(
            'itterative/danbooru_wikis_full',
            filename='tags.parquet',
            repo_type='dataset',
            revision='5261235d60fd4be1809672da3c099c0b4dd3c586',
        )

        wiki_csv = huggingface_hub.hf_hub_download(
            'itterative/danbooru_wikis_full',
            filename='wiki_pages.parquet',
            repo_type='dataset',
            revision='5261235d60fd4be1809672da3c099c0b4dd3c586',
        )
    except:
        traceback.print_exc()
        raise AssertionError('Could not grab necessary files for building the wiki')
    
    connection.create_function('dtext_to_markdown', dtext_to_markdown)
    connection.create_function('dtext_to_raw', dtext_to_raw)

    yield 0.33, 'Ingesting wiki data'

    connection.execute(f"create temporary table wiki_csv as select id, title, body from '{wiki_csv}' where starts_with(title, 'api:') = false and starts_with(title, 'howto:') = false")
    connection.execute(f"create temporary table tags_csv as select name, max(post_count) as post_count from '{tags_csv}' group by name")

    wiki_page_count = int(connection.sql('select count() from wiki_csv').fetchone()[0])
    wiki_page_batch_size = int(wiki_page_count * 0.5) // 100

    for offset in tqdm.tqdm(range(0, wiki_page_count, wiki_page_batch_size), desc='parsing wiki pages'):
        # gradio tends to timeout when ingesting too much at a time
        # (likely cause duckdb holds the GIL somehow)

        connection.execute(f"""
            insert into wiki (id, post_count, title, markdown, search_title, search_text) 
                select
                    wiki_csv.id as id,
                    greatest(tags_csv.post_count, 1) as post_count,
                    wiki_csv.title as title,
                    dtext_to_markdown(wiki_csv.body) as markdown,
                    list_reduce(regexp_split_to_array(lower(wiki_csv.title), '[^a-z0-9]+'), (acc, x) -> concat(acc, ' ', x)) as search_title,
                    list_reduce(regexp_split_to_array(lower(dtext_to_raw(wiki_csv.body)), '[^a-z0-9]+'), (acc, x) -> concat(acc, ' ', x)) as search_text
                from (select * from wiki_csv limit ? offset ?) wiki_csv left join tags_csv on wiki_csv.title = tags_csv.name
        """, parameters=(wiki_page_batch_size, offset))

        yield 0.33, f"Ingesting wiki data: {(min(wiki_page_count, offset+wiki_page_batch_size))*100/wiki_page_count:.2f}%"

    yield 0.75, 'Creating wiki index'

    connection.execute("pragma create_fts_index('wiki', 'id', 'search_title', 'search_text', stemmer = 'english')")

    connection.remove_function('dtext_to_markdown')
    connection.remove_function('dtext_to_raw')
