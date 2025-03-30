import pytest

@pytest.mark.parametrize(
    [
        'initial_tags',
        'edited_tags',
        'new_tags',
        'wanted_tags',
    ],
    [
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_3, TAG_4',
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_3, TAG_4',
        ],
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2',
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2',
        ],
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_4',
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_4',
        ],
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, TAG_2, BREAK, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, TAG_2, BREAK, TAG_3',
        ],
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, TAG_2, BREAK, TAG_3',
            'TAG_1, TAG_4, BREAK, TAG_2, TAG_3',
            'TAG_1, TAG_2, TAG_4, BREAK, TAG_3',
        ],
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2, TAG_3, TAG_4, TAG_5',
            'TAG_1, BREAK, TAG_2, TAG_3, TAG_4, TAG_5',
        ],
        [
            'TAG_1, BREAK, TAG_2, TAG_3',
            'TAG_1, BREAK, TAG_2',
            'TAG_1, BREAK, TAG_2, TAG_3, TAG_4, TAG_5',
            'TAG_1, BREAK, TAG_2, TAG_4, TAG_5',
        ],
    ],
    ids=[
        'add new tag',
        'remove old tag',
        'add and remove tag',
        'move tag',
        'insert tag',
        'append tags',
        'remove and append tags'
    ]
)
def test_post_process_manual_edits(
        initial_tags: str,
        edited_tags: str,
        new_tags: str,
        wanted_tags: str,
):
    from yadt.process_prediction import post_process_manual_edits

    results = post_process_manual_edits(initial_tags, edited_tags, new_tags)
    assert results == wanted_tags
