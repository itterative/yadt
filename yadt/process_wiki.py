import tqdm
import typing
import traceback
import warnings

import re
import pathlib
import duckdb
import dataclasses
import huggingface_hub

def _wiki_processors():
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
        regex = re.compile(pattern, flags=re_flags)

        def _apply_regex(fn):
            def _wrapper(text: str) -> str:
                return re.sub(regex, lambda m: fn(*m.groups(), **kwargs), text)

            return _wrapper

        return _apply_regex
    
    def apply_regex_node(pattern: str, flags = 0, **kwargs):
        regex = re.compile(pattern, flags=flags)

        def _update_node(root: Node, node: Node):
            for node_child in node.children:
                _update_node(root, node_child)

            node.start_pos = node.start_pos + root.start_pos
            node.end_pos = node.start_pos + root.start_pos

        def _apply_regex(fn):
            def _wrapper(node: Node) -> list[Node]|None:
                if node.type != 'string':
                    if node.tag in ('nodtext', 'url', 'code'):
                        return None

                    has_change = False
                    for node_child in list(node.children):
                        nodes = _wrapper(node_child)

                        if nodes is None:
                            continue

                        has_change = True
                        i_child = node.children.index(node_child)
                        node.children = node.children[:i_child] + nodes + node.children[i_child+1:]
                
                    return [node] if has_change else None


                nodes: list[Node] = []
                content = node.content

                current_pos = 1

                for match in re.finditer(regex, content):
                    match_start, match_end = match.span()

                    new_node: Node|None = fn(*match.groups(), **kwargs)
                    if new_node is None:
                        continue

                    if match_start > 0:
                        nodes.append(Node(type='string', content=content[current_pos-1:match_start], start_pos=current_pos-1, end_pos=match_start))
                    
                    nodes.append(new_node)

                    current_pos = match_end + 1

                if current_pos == 1:
                    return None

                if current_pos <= len(content):
                    nodes.append(Node(type='string', content=content[current_pos-1:], start_pos=current_pos, end_pos=len(content)-1))

                for new_node in nodes:
                    _update_node(node, new_node)

                # print('apply_regex_node', fn.__name__, len(nodes))
                # print_tag_tree(Node(type='tag', children=nodes))
                # print('')

                return nodes

            return _wrapper

        return _apply_regex
    
    def print_tag_tree(node: 'Node', indent: int = 0):
        if node.type == 'tag':
            # print(' ' * indent + f'tag: tag={node.tag} content={repr(node.content)} attr={repr(node.attr)} start={node.start_pos} end={node.end_pos} children={len(node.children)}')
            print(' ' * indent + f'tag: tag={node.tag} attr={repr(node.attr)}')
        else:
            # print(' ' * indent + f'string: content={repr(node.content[:40])} ({len(node.content)}) start={node.start_pos} end={node.end_pos}')
            print(' ' * indent + f'string: content={repr(node.content[:40])} ({len(node.content)})')

        for _node in node.children:
            print_tag_tree(_node, indent=indent+2)


    @dataclasses.dataclass
    class Node:
        type: typing.Literal['tag', 'string']
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
            'table', 'colgroup',
        ]

        dtext_tag_one_line = [
            'b', 'i', 'u', 's'
        ]

        dtext_tags_with_attr = [
            'thead', 'tbody', 'tr', 'col', 'th', 'td',
            'expand',
            'url',
        ]

        dtext_tags_with_no_parsing = [
            'nodtext', 'code',
        ]

        tags = dtext_tags + dtext_tags_with_attr

        re_separators_first = separators[0] if separators[0] != '[' else f'\\{separators[0]}'
        re_separators_second = separators[1] if separators[1] != ']' else f'\\{separators[1]}'

        dtext_split_parts = []
        for tag in dtext_tags:
            dtext_split_parts.append(f'{re_separators_first}{tag}{re_separators_second}')
            dtext_split_parts.append(f'{re_separators_first}/{tag}{re_separators_second}')
        for tag in dtext_tags_with_attr:
            dtext_split_parts.append(f'{re_separators_first}{tag}.*?{re_separators_second}')
            dtext_split_parts.append(f'{re_separators_first}/{tag}{re_separators_second}')

        dtext_split_re = re.compile('(' + '|'.join(dtext_split_parts) + ')')

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
                    node.attr = _parse_attr(node.content)
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

    def format_tag_tree(node: Node, raw_text: bool = False, second_pass: bool = False):
        def _format_dtext(node: Node):
            if node.type == 'string':
                return _format_string(node)

            match node.tag:
                case 'code':
                    return _format_code(node)
                case 'table':
                    return _format_table(node)
                case 'spoilers':
                    return _format_spoilers(node)
                case 'expand':
                    return _format_expand(node)
                case 'ul':
                    return _format_list(node)
                case _:
                    return _format_dtext_simple(node)

        def _format_dtext_simple(node: Node):
            if node.type == 'string':
                return _format_string(node)

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
                case 'h':
                    return _format_dtext_heading(node)
                case 'tn':
                    return _format_dtext_note(node)
                case 'url':
                    return _format_dtext_url(node)
                case 'br':
                    return _format_dtext_breakline(node)
                case 'hr':
                    return _format_dtext_headline(node)
                case _:
                    raise AssertionError(f'unsupported simple dtext: {node.tag}')


        def _format_table(node: Node):
            if node.type == 'string':
                return '\n' # ignore whatever strings
            
            match node.tag:
                case 'table':
                    return '<table>' + ''.join(map(_format_table, node.children)) + '</table>'
                case 'thead':
                    return ''.join(map(_format_table, node.children))
                case 'tbody':
                    return ''.join(map(_format_table, node.children))
                case 'tr':
                    return '<tr>' + ''.join(map(_format_table, node.children)) + '</tr>'
                case 'th':
                    return '<th>' + ''.join(map(_format_dtext_simple, node.children)) + '</th>'
                case 'td':
                    return '<td>' + ''.join(map(_format_dtext_simple, node.children)) + '</td>'
                case _:
                    raise AssertionError(f'unsupported table dtext: {node.tag}')

        def _format_list(node: Node):
            if node.type == 'string':
                return _format_string(node)
            
            match node.tag:
                case 'ul':
                    return '<ul>\n' + '\n'.join(map(_format_list, node.children)) + '\n</ul>'
                case 'li':
                    return '<li>' + ''.join(map(_format_dtext_simple, node.children)) + '</li>'
                case _:
                    raise AssertionError(f'unsupported list dtext: {node.tag}')


        def _format_string(node: Node):
            return node.content

        def _format_nodtext(node: Node):
            assert len(node.children) <= 1, f"expected one or less children for nodtext"
            assert all(map(lambda n: n.type == 'string', node.children)), f"expected childred to be string for nodtext; got: {', '.join(map(lambda n: n.type if n.type == 'string' else n.type + ':' + n.tag, node.children))}"

            if second_pass:
                return ''.join(map(lambda n: n.content, node.children))

            return '[nodtext]' + ''.join(map(lambda n: n.content, node.children)) + '[/nodtext]'
        
        def _format_code(node: Node):
            assert len(node.children) <= 1, f"expected one or less children for code"
            assert all(map(lambda n: n.type == 'string', node.children)), f"expected childred to be string for code; got: {', '.join(map(lambda n: n.type if n.type == 'string' else n.type + ':' + n.tag, node.children))}"

            text = ''.join(map(lambda n: n.content, node.children))
            return f'```\n{text.removesuffix("\n")}\n```'


        # no markdown equivalent
        def _format_spoilers(node: Node):
            return ''.join(map(_format_dtext, node.children))
        
        # no markdown equivalent
        def _format_expand(node: Node):
            text = ''.join(map(_format_dtext, node.children))
            title = node.attr.get('expand', None)
            if title is not None:
                text = f'<h5>{title}</h5>\n<hr>\n{text}'
            return text


        def _format_quote(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return f'<blockquote>{text}</blockquote>'

        def _format_dtext_bold(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join([f'<b>{t.strip()}</b>' for t in text.splitlines() if len(t) > 0])
        
        def _format_dtext_italic(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join([f'<i>{t.strip()}</i>' for t in text.splitlines() if len(t) > 0])
        
        def _format_dtext_underline(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join([f'<ins>{t.strip()}</ins>' for t in text.splitlines() if len(t) > 0])
        
        def _format_dtext_strikethrough(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join([f'<s>{t.strip()}</s>' for t in text.splitlines() if len(t) > 0])
        
        def _format_dtext_heading(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            heading = 'h' + node.attr['type']
            id = node.attr.get('id')
            if id is not None:
                return f'<{heading} id="{id}">{text}</{heading}>'    

            return f'<{heading}>{text}</{heading}>'
        
        def _format_dtext_note(node: Node):
            text = ''.join(map(_format_dtext_simple, node.children))
            return '\n'.join([f'<sub><sup>{t.strip()}</sub></sup>' for t in text.splitlines() if len(t) > 0])
        
        def _format_dtext_url(node: Node):
            assert len(node.children) <= 1, f"expected one or less children for code"
            assert all(map(lambda n: n.type == 'string', node.children)), f"expected childred to be string for code; got: {', '.join(map(lambda n: n.type if n.type == 'string' else n.type + ':' + n.tag, node.children))}"

            title = ''.join(map(lambda n: n.content, node.children))
            url = title

            url_attr = node.attr.get('url')
            if url_attr is not None:
                url = url_attr

            return f'<a href="{url}">{title}</a>'
        
        def _format_dtext_breakline(node: Node):
            return '<br>\n'
        
        def _format_dtext_headline(node: Node):
            return '\n<hr>\n'

        def _format_string_node_only(node: Node):
            if node.type == 'string':
                return node.content

            return ''.join(map(_format_string_node_only, node.children))
        

        if raw_text:
            return _format_string_node_only(node)

        return ''.join(map(_format_dtext, node.children))

    @apply_regex_node('\\[nodtext\\](.*?)\\[/nodtext\\]')
    def parse_nodtext(text: str):
        child = Node(type='string', content=text)
        return Node(type='tag', tag='nodtext', children=[child])

    @apply_regex_node('h([0-9])(#[^\\."]+|)\\.(.+)')
    def parse_heading(heading_type: str, anchor: str, rest: str):
        child = Node(type='string', content=rest.strip())

        if len(anchor) > 1:
            return Node(type='tag', tag='h', attr={'type': heading_type, 'id': anchor[1:]}, children=[child])

        return Node(type='tag', tag='h', attr={'type': heading_type}, children=[child])

    @apply_regex_node('(?=\n|^)((?:[*]+\\s?.+?(?:\n|$))+)(?=\n|$)', flags=re.RegexFlag.MULTILINE)
    def parse_list_items(text: str):
        level = 1

        node = Node(type='tag', tag='ul', end_pos=len(text)-1)
        ul_node = node

        # TODO: this should put other <ul> in <li> not outside
        current_pos = 0

        for line in text.splitlines():
            # why would \x0b be counted as a separate empty line?
            if len(line.strip()) == 0:
                continue

            line_len_before = len(line)
            line_after = line.lstrip('*')
            line_len_after = len(line_after)

            line_level = line_len_before - line_len_after
            assert line_level > 0, f"unexpected line level {line_level}: {repr(text)}"

            for _ in range(level, line_level):
                _node = Node(type='tag', tag='ul', start_pos=current_pos, parent=node)
                node.children.append(_node)
                node = _node

            for _ in range(line_level, level):
                node.end_pos = current_pos-1
                node = node.parent
                assert node is not None, f"unexpected non-existing parent: {repr(text)}"

            node.children.append(Node(type='tag', tag='li', start_pos=current_pos+line_level, end_pos=current_pos+len(line)-1, children=[Node(type='string', content=line_after.strip())]))

            level = line_level
            current_pos += len(line)

        return ul_node


    @apply_regex_node('([a-zA-Z0-9]*)\\[\\[(.+?)\\]\\]([a-zA-Z0-9]*)', links_qualifier = re.compile('\\(.+\\)'))
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
            child = Node(type='string', content=left + parts[0].strip() + right)
            return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/wiki_pages/{parts[0].lower().replace(" ", "_")}{anchor}'}, children=[child])
        elif parts[1] == '':
            child = Node(type='string', content=left + re.sub(links_qualifier, '', parts[0]).strip() + right)
            return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/wiki_pages/{parts[0].strip().lower().replace(" ", "_")}{anchor}'}, children=[child])
        else:
            child = Node(type='string', content=left + parts[1].strip() + right)
            return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/wiki_pages/{parts[0].strip().lower().replace(" ", "_")}{anchor}'}, children=[child])
        
    @apply_regex_node('\\{\\{(.+?)\\}\\}', links_qualifier = re.compile('\\(.+\\)'))
    def parse_tag_search_links(contents: str, links_qualifier: re.Pattern = None):
        parts = contents.split('|', 1)

        if len(parts) == 1:
            child = Node(type='string', content=parts[0].strip())
            return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/posts?tags={parts[0].lower()}'}, children=[child])
        elif parts[1] == '':
            child = Node(type='string', content=re.sub(links_qualifier, '', parts[0]).strip())
            return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/posts?tags={parts[0].lower()}'}, children=[child])
        else:
            child = Node(type='string', content=parts[1].strip())
            return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/posts?tags={parts[0].strip().lower()}'}, children=[child])

    @apply_regex_node('(.|)(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{2,256}\\.[a-z]{2,4}\\b(?:[-a-zA-Z0-9@:%_\\+.~#?&//=]*))(.|)')
    def parse_links_raw(lookback: str, url: str, lookahead: str):
        if lookback in ('"', '>') and lookahead in ('"', '<'):
            return None

        child = Node(type='string', content=url)
        return Node(type='tag', tag='url', children=[child])

    @apply_regex_node('"([^"]+?)":\\[(.+?)\\]')
    def parse_links_alt(title: str, url: str):
        child = Node(type='string', content=title)
        return Node(type='tag', tag='url', attr={'url': url}, children=[child])

    @apply_regex_node('"([^"]+?)":(\\[.+?\\])')
    def parse_links_alt2(title: str, url: str):
        child = Node(type='string', content=title)
        return Node(type='tag', tag='url', attr={'url': url}, children=[child])
    
    @apply_regex_node('"([^"]+?)":(?=http|/|#)([^\\s]+)')
    def parse_links_alt3(title: str, url: str):
        child = Node(type='string', content=title)
        return Node(type='tag', tag='url', attr={'url': url}, children=[child])
    
    @apply_regex_node('\\[(http.+?)\\]\\((.+?)\\)')
    def parse_links_md(url: str, title: str):
        child = Node(type='string', content=title)
        return Node(type='tag', tag='url', attr={'url': url}, children=[child])

    @apply_regex_node('\\[(?!http)(.+?)\\]\\((.+?)\\)')
    def parse_links_md_reversed(title: str, url: str):
        child = Node(type='string', content=title)
        return Node(type='tag', tag='url', attr={'url': url}, children=[child])

    @apply_regex_node('<a href="(.+?)">(.+?)</a>')
    def parse_links_alt_html(url: str, title: str):
        child = Node(type='string', content=title)
        return Node(type='tag', tag='url', attr={'url': url}, children=[child])

    @apply_regex_node('<(http.+?)>')
    def parse_links_alt_htmlish(url: str):
        child = Node(type='string', content=url)
        return Node(type='tag', tag='url', content=url, children=[child])
    
    @apply_regex_node('(.|)@([a-z0-9A-Z-_]+)(.|)')
    def parse_links_user(lookback: str, user: str, lookahead: str):
        # TODO:@ inside another link (test)
        #      idk, this might need to create a tree instead of just using regex

        if lookback in ('"', '>') and lookahead in ('"', '<'):
            return None

        child = Node(type='string', content=f'@{user}')
        return Node(type='tag', tag='url', attr={'url': f'https://danbooru.donmai.us/users?name={user}'}, children=[child])


    @apply_regex_node('\\[br\\]')
    def parse_breaks():
        return Node(type='tag', tag='br')

    @apply_regex_node('\\[hr\\]')
    def parse_breakslines():
        return Node(type='tag', tag='hr')


    @apply_tag_tree(('[', ']'))
    def parse_dtext_bbcode_style(tree: Node, raw_text: bool = False, second_pass: bool = False):
        return format_tag_tree(tree, raw_text=raw_text, second_pass=second_pass)

    @apply_tag_tree(('<', '>'))
    def parse_dtext_html_style(tree: Node, raw_text: bool = False, second_pass: bool = False):
        return format_tag_tree(tree, raw_text=raw_text, second_pass=second_pass)

    def parse_dtext_second_pass(dtext: str, raw_text: bool = False):
        parsers = [
            parse_nodtext,
            parse_breaks,
            parse_breakslines,

            parse_heading,
            parse_list_items,

            parse_links_alt,
            parse_links_alt2,
            parse_links_alt3,
            parse_links_md,
            parse_links_md_reversed,
            # parse_links_alt_html,
            parse_links_alt_htmlish,
            parse_links_raw,

            parse_links,
            parse_tag_search_links,
            parse_links_user,
        ]

        root = Node(type='tag')
        nodes = [Node(type='string', content=dtext, start_pos=0, end_pos=len(dtext)-1, parent=root)]

        for parser in parsers:
            for _node in list(nodes):
                _nodes = parser(_node)

                if _nodes is None:
                    continue

                i_node = nodes.index(_node)
                nodes = nodes[:i_node] + _nodes + nodes[i_node+1:]

        root.children = nodes

        # print('------')
        # print_tag_tree(root)

        return format_tag_tree(root, raw_text=raw_text, second_pass=True)

    def parse_dtext(dtext: str, raw_text: bool = False) -> str:
        if dtext is None:
            return ''

        dtext = dtext.replace('\r\n', '\n')

        # TODO: convert html style to bbcode style
        # dtext = parse_dtext_html_style(dtext, raw_text=raw_text)
        dtext = parse_dtext_bbcode_style(dtext, raw_text=raw_text)
        dtext = parse_dtext_second_pass(dtext, raw_text=raw_text)

        return dtext

    def dtext_to_markdown(dtext: str) -> str:
        return parse_dtext(dtext)

    def dtext_to_raw(dtext: str) -> str:
        return parse_dtext(dtext, raw_text=True)
    
    return dtext_to_markdown, dtext_to_raw


def process_wiki(connection: duckdb.DuckDBPyConnection):
    dtext_to_markdown, dtext_to_raw = _wiki_processors()

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

    with warnings.catch_warnings(action='ignore'):
        connection.create_function('dtext_to_markdown', dtext_to_markdown)
        connection.create_function('dtext_to_raw', dtext_to_raw)

    yield 0.33, 'Ingesting wiki data'

    connection.execute(f"create temporary table tags_csv as select name, max(post_count) as post_count from '{tags_csv}' group by name")

    connection.execute(f"""
        create temporary table wiki_csv as select id, title, body from '{wiki_csv}'
            where starts_with(title, 'api:') = false and starts_with(title, 'howto:') = false and starts_with(title, 'template:') = false and starts_with(title, 'help:') = false
    """)

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
