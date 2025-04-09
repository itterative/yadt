import pytest
import duckdb

from yadt.db_wiki import WikiDB
from yadt.process_wiki import _wiki_processors, process_wiki

@pytest.fixture
def wiki_db(injector):
    yield injector.get(WikiDB)

def test_process_wiki(wiki_db: WikiDB):
    # making sure the processing works
    with wiki_db._conn() as connection:
        for _ in process_wiki(connection): pass

def _test_dtext_to_markdown(dtext: str, markdown: str):
    dtext_to_markdown, _ = _wiki_processors()

    dtext = dtext.lstrip('\n').rstrip(' \t\n')
    markdown = markdown.lstrip('\n').rstrip(' \t\n')
    result = dtext_to_markdown(dtext)

    print('')
    print('----- wanted')
    print(markdown)
    print('----- got')
    print(result)
    print('-----')
    print('')

    assert result == markdown

@pytest.mark.parametrize(
    [
        'dtext',
        'markdown',
    ],
    [
        [ '[b]bold[/b]', '<b>bold</b>' ],
        [ '[i]italics[/i]', '<i>italics</i>' ],
        [ '[u]underline[/u]', '<ins>underline</ins>' ],
        [ '[s]strikethrough[/s]', '<s>strikethrough</s>' ],
        [ '[tn]note[/tn]', '<sub><sup>note</sub></sup>' ],
        [ '[spoilers]ssh![/spoilers]', 'ssh!' ],
        [ '[nodtext][u]nodtext[/u][/nodtext]', '[u]nodtext[/u]' ],
        [ '[code][u]code[/u][/code]', '```\n[u]code[/u]\n```' ],
        [ 'line[br]break', 'line<br>\nbreak' ],
    ],
    ids=[
        'bold',
        'italics',
        'underline',
        'strikethrough',
        'note',
        'spoilers',
        'nodtext',
        'code',
        'linebreak',
    ]
)
def test_dtext_to_markdown_basic_formatting(dtext: str, markdown: str):
    _test_dtext_to_markdown(dtext, markdown)

@pytest.mark.parametrize(
    [
        'dtext',
        'markdown',
    ],
    [
        [
            'https://danbooru.donmai.us',
            '<a href="https://danbooru.donmai.us">https://danbooru.donmai.us</a>'
        ],
        [
            '<https://danbooru.donmai.us>',
            '<a href="https://danbooru.donmai.us">https://danbooru.donmai.us</a>'
        ],
        [
            '"Danbooru":[https://danbooru.donmai.us]',
            '<a href="https://danbooru.donmai.us">Danbooru</a>'
        ],
        [
            '[Danbooru](https://danbooru.donmai.us)',
            '<a href="https://danbooru.donmai.us">Danbooru</a>'
        ],
        [
            '[https://danbooru.donmai.us](Danbooru)',
            '<a href="https://danbooru.donmai.us">Danbooru</a>'
        ],
        [
            '<a href="https://danbooru.donmai.us">Danbooru</a>',
            '<a href="https://danbooru.donmai.us">Danbooru</a>'
        ],
        [
            '[url]https://danbooru.donmai.us[/url]',
            '<a href="https://danbooru.donmai.us">https://danbooru.donmai.us</a>'
        ],
        [
            '[url=https://danbooru.donmai.us]Danbooru[/url]',
            '<a href="https://danbooru.donmai.us">Danbooru</a>'
        ],
        [
            '"ToS":[/terms_of_service]',
            '<a href="/terms_of_service">ToS</a>'
        ],
        [
            '"Here":[#dtext-basic-formatting]',
            '<a href="#dtext-basic-formatting">Here</a>'
        ],
        [
            '[[Kantai Collection]]',
            '<a href="https://danbooru.donmai.us/wiki_pages/kantai_collection">Kantai Collection</a>'
        ],
        [
            '[[Kantai Collection#External-links]]',
            '<a href="https://danbooru.donmai.us/wiki_pages/kantai_collection#dtext-external-links">Kantai Collection</a>'
        ],
        [
            '[[Kantai Collection|Kancolle]]',
            '<a href="https://danbooru.donmai.us/wiki_pages/kantai_collection">Kancolle</a>'
        ],
        [
            '[[Fate (series)|]]',
            '<a href="https://danbooru.donmai.us/wiki_pages/fate_(series)">Fate</a>'
        ],
        [
            '[[cat]]s, 19[[60s]]',
            '<a href="https://danbooru.donmai.us/wiki_pages/cat">cats</a>, <a href="https://danbooru.donmai.us/wiki_pages/60s">1960s</a>'
        ],
        [
            '{{kantai_collection comic}}',
            '<a href="https://danbooru.donmai.us/posts?tags=kantai_collection comic">kantai_collection comic</a>'
        ],
        [
            '{{kantai_collection comic|Kancolle Comics}}',
            '<a href="https://danbooru.donmai.us/posts?tags=kantai_collection comic">Kancolle Comics</a>'
        ],
        [
            '@evazion',
            '<a href="https://danbooru.donmai.us/users?name=evazion">@evazion</a>'
        ]
    ],
    ids=[
        'basic link',
        'basic link with delimiters',
        'link with custom text',
        'markdown style',
        'reverse markdown style',
        'html style',
        'bbcode style',
        'bbcode style with custom text',
        'link to a danbooru page',
        'link to a specific section of the current page',
        'link to a wiki',
        'link to a specific section of a wiki article',
        'link to a wiki with custom text',
        'link to a wiki without the qualifier',
        'link with adjacent text becomes part of link',
        'link to a tag search',
        'link to a tag search with custom text',
        'link to a user',
    ]
)
def test_dtext_to_markdown_links(dtext: str, markdown: str):
    _test_dtext_to_markdown(dtext, markdown)

@pytest.mark.parametrize(
    [
        'dtext',
        'markdown',
    ],
    [
        [ 'h1. testing', '<h1>testing</h1>' ],
        [ 'h3. testing', '<h3>testing</h3>' ],
        [ '[quote]\nquote\n[/quote]', '<blockquote>\nquote\n</blockquote>' ],
        [ '[quote]\nquote\n[quote]\nquote\n[/quote]\nquote\n[/quote]', '<blockquote>\nquote\n<blockquote>\nquote\n</blockquote>\nquote\n</blockquote>' ],
        [ '[expand]expand[/expand]', 'expand' ],
        [ '[expand=title]expand[/expand]', '<h5>title</h5>\n<hr>\nexpand' ],
    ],
    ids=[
        'heading h1',
        'heading h3',
        'quote',
        'quote in quote',
        'expand',
        'expand with title',
    ]
)
def test_dtext_to_markdown_simple(dtext: str, markdown: str):
    _test_dtext_to_markdown(dtext, markdown)


@pytest.mark.parametrize(
    [
        'dtext',
        'markdown',
    ],
    [
        [
            '''
*item1
*item2
            ''',
            '''
<ul>
<li>item1</li>
<li>item2</li>
</ul>
            '''
        ],
        [
            '''
*item1
**subitem1
*item2
            ''',
            '''
<ul>
<li>item1</li>
<ul>
<li>subitem1</li>
</ul>
<li>item2</li>
</ul>
            '''
        ],
    ],
    ids=[
        'basic',
        'with subitems',
    ]
)
def test_dtext_to_markdown_list(dtext: str, markdown: str):
    _test_dtext_to_markdown(dtext, markdown)


@pytest.mark.parametrize(
    [
        'dtext',
        'markdown',
    ],
    [
        [
            '''
[table]
[thead]
[tr]
[th]col 1[/th]
[th]col 2[/th]
[th]col 3[/th]
[/tr]
[/thead]

[tbody]
[tr]
[td]value 1[/td]
[td]value 2[/td]
[td]value 3[/td]
[/tr]

[tr]
[td]value 1[/td]
[td]value 2[/td]
[td]value 3[/td]
[/tr]
[/tbody]
[/table]
            ''',
            '''
<table>

<tr>
<th>col 1</th>
<th>col 2</th>
<th>col 3</th>
</tr>


<tr>
<td>value 1</td>
<td>value 2</td>
<td>value 3</td>
</tr>
<tr>
<td>value 1</td>
<td>value 2</td>
<td>value 3</td>
</tr>

</table>
            '''
        ],
        [
            '''
[expand=expand section]
* 1. "example":#dtext-example
* 2. "'some' example":#dtext-some-example
[/expand]
            ''',
            '''
<h5>expand section</h5>
<hr>

<ul>
<li>1. <a href="#dtext-example">example</a></li>
<li>2. <a href="#dtext-some-example">'some' example</a></li>
</ul>

            '''
        ],
        [
            '''
*[url=http://example.com]example[/url]
            ''',
            '''
<ul>
<li><a href="http://example.com">example</a></li>
</ul>
            '''
        ],
    ],
    ids=[
        'simple table',
        'links in list items',
        'bbcode links in list items',
    ]
)
def test_dtext_to_markdown_complex(dtext: str, markdown: str):
    _test_dtext_to_markdown(dtext, markdown)


@pytest.mark.parametrize(
    [
        'dtext',
        'markdown',
    ],
    [
        [ '[nodtext]<https://danbooru.donmai.us>[/nodtext]', '<https://danbooru.donmai.us>' ],
        [ '[nodtext]h4. test[/nodtext]', 'h4. test' ],
        [ '[nodtext][b]test[/b][/nodtext]', '[b]test[/b]' ],
        [ '"example":#dtext-example', '<a href="#dtext-example">example</a>' ],
        [ '"example":/test', '<a href="/test">example</a>' ],
        [ '"example":http://example.com', '<a href="http://example.com">example</a>' ],
        [ 'h3#heading-with-id. heading with id', '<h3 id="heading-with-id">heading with id</h3>' ],
        [ '"@example":http://example.com', '<a href="http://example.com">@example</a>' ],
        [ '[[example|ex@mple]]', '<a href="https://danbooru.donmai.us/wiki_pages/example">ex@mple</a>' ],
        [ '([[idolmaster_cinderella_girls| iDOLM@STER Cinderella Girls]])', '(<a href="https://danbooru.donmai.us/wiki_pages/idolmaster_cinderella_girls">iDOLM@STER Cinderella Girls</a>)' ],
        [
            '"example1":http://example.com (some text ""quoted example":[http://example.com]" more text)',
            '<a href="http://example.com">example1</a> (some text "<a href="http://example.com">quoted example</a>" more text)'
        ],
    ],
    ids=[
        'nodtext parsing for links',
        'nodtext parsing for headings',
        'nodtext parsing for dtext',
        'dtext anchor without brackets',
        'relative link without brackets',
        'link without brackets',
        'heading with id',
        "@user, but with link",
        "@ inside another link",
        "@ inside another link 2",
        "weird linking",
    ]
)
def test_dtext_to_markdown_edge_cases(dtext: str, markdown: str):
    _test_dtext_to_markdown(dtext, markdown)
