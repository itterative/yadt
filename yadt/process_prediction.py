from typing import Tuple, Dict, List

import numpy as np

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}

def post_process_prediction(
        rating: Dict[str, float],
        general_res: Dict[str, float],
        character_res: Dict[str, float],
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        replace_underscores: bool,
        trim_general_tag_dupes: bool,
        escape_brackets: bool,
        prefix_tags: str = None,
        keep_tags: str = None,
        ban_tags: str = None,
        map_tags: str = None,
):
    def _threshold(tags: List[Tuple[str, float]], t: float, mcut: bool):
        def mcut_threshold(probs):
            """
            Maximum Cut Thresholding (MCut)
            Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
            for Multi-label Classification. In 11th International Symposium, IDA 2012
            (pp. 172-183).
            """
            sorted_probs = probs[probs.argsort()[::-1]]
            difs = sorted_probs[:-1] - sorted_probs[1:]
            t = difs.argmax()
            thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
            return thresh

        if mcut:
            probs = np.array([x[1] for x in tags])
            t = max(t, mcut_threshold(probs))
        
        return [x for x in tags if x[1] >= t]

    def _replace_underscore(tags: List[Tuple[str, float]]):
        if not replace_underscores:
            return tags

        return [
            [ tag.replace('_', ' ') if tag not in kaomojis else tag, prob] for tag, prob in tags
        ]

    def _generate_string(character_res: List[Tuple[str, float]], general_res: List[Tuple[str, float]]):
        character_res = character_res
        general_res = list(map(lambda x: (x[0], x[1] - 1.0), general_res))

        # print('generate_string', character_res, general_res)

        sorted_tags = sorted(character_res + general_res, key=lambda x: x[1], reverse=True)
        sorted_tags = _map_tokens(sorted_tags)
        sorted_tags = _keep_tokens(sorted_tags)
        sorted_tags = _prefix_tokens(sorted_tags)
        sorted_tags = _ban_tokens(sorted_tags)
        sorted_tags = sorted(sorted_tags, key=lambda x: x[1], reverse=True)

        sorted_tags = list(map(lambda x: x[0], sorted_tags))
        generated_string = ', '.join(sorted_tags)

        if escape_brackets:
            generated_string = generated_string.replace("(", "\\(").replace(")", "\\)")

        return generated_string
    
    def _trim_general_tag_dupes(tags: List[Tuple[str, float]]):
        if not trim_general_tag_dupes:
            return tags

        def matches_tag(search_tag):
            # returns True if the search_tag's words are included in tag's words
            #   example: 'A B' & 'A' -> True
            #   example: 'Aa B' & 'A' -> False
            #   example: 'A B C' & 'B C' -> True
            #   example: 'A B Cc' & 'B C' -> False

            search_tag = search_tag[0]

            def _matches_tag(tag):
                tag = tag[0]

                if tag == search_tag:
                    return False

                search_tag_words = search_tag.split()
                tag_words = tag.split()
                len_tag_words = len(tag_words)

                for i in range(len(search_tag_words)-len_tag_words+1):
                    if tag_words == search_tag_words[i:len_tag_words+i]:
                        return True

                return False

            return _matches_tag
        
        tags_new = list(tags)

        removed = True
        while removed:
            removed = False
            for i in range(len(tags_new)):
                tag = tags_new[i]
                found_tags = list(filter(matches_tag(tag), tags_new))

                if len(found_tags) == 0:
                    continue

                for found_tag in found_tags:
                    tags_new.remove(found_tag)
                    break

                removed = True
                break

        return tags_new

    def _prefix_tokens(tags: List[Tuple[str, float]]):
        if prefix_tags is None:
            return tags
        
        tags_new: List[Tuple[str, float]] = []

        prefix_tags_list = list(filter(lambda t: len(t) > 0, map(lambda t: t.strip(), prefix_tags.split(','))))
        max_prob = max(([0] + list(map(lambda t: t[1], tags))))

        for i, tag in enumerate(reversed(prefix_tags_list)):
            tags_new.append((tag, i + 1.0 + max_prob))

        tags_new.extend(tags)
        # print('prefix_tokens', tags_new)
        return tags_new

    def _keep_tokens(tags: List[Tuple[str, float]]):
        if keep_tags is None:
            return tags

        tags_new: List[Tuple[str, float]] = []

        keep_tags_list = list(filter(lambda t: len(t) > 0, map(lambda t: t.strip(), keep_tags.split(','))))
        max_prob = max(([0] + list(map(lambda t: t[1], tags))))

        for tag, prob in tags:
            if tag in keep_tags_list:
                prob = 5.0 + (1.0 - keep_tags_list.index(tag) / len(keep_tags_list))

            tags_new.append((tag, prob))
        
        tags_new.append(('BREAK', max_prob + 1.0))
        # print('keep_tokens', tags_new)
        return tags_new

    def _ban_tokens(tags: List[Tuple[str, float]]):
        if ban_tags is None:
            return tags
        
        tags_new: List[Tuple[str, float]] = []

        ban_tags_list = list(map(lambda t: t.strip(), ban_tags.split(',')))

        for tag, prob in tags:
            if tag in ban_tags_list:
                continue

            tags_new.append((tag, prob))

        # print('ban_tokens', tags_new)
        return tags_new


    def _map_tokens(tags: List[Tuple[str, float]]):
        if map_tags is None:
            return tags

        import re
        line_re = re.compile("^(\\s*|(.+):(.+))$")

        assert all(map(lambda s: line_re.match(s) is not None, map_tags.splitlines())), "Map tokens is not valid: expected lines of format: token, token, ... : token"

        map_tags_dict: Dict[str, List[str]] = {}

        for line in map_tags.splitlines():
            _, tokens, to_token = line_re.match(line).groups()

            if to_token is None:
                continue

            to_token = to_token.strip()
            to_token = list(map(lambda t: t.strip(), to_token.split(',')))

            for token in tokens.split(','):
                map_tags_dict[token.strip()] = to_token

        # print('map_tags_dict', map_tags_dict)

        tags_old = list(tags)

        has_mapped_a_tag = True
        for _ in range(20):
            has_mapped_a_tag = False

            tags_new: List[Tuple[str, float]] = []

            for tag, prob in tags_old:
                mapped_tags = map_tags_dict.get(tag, None)

                if mapped_tags is None:
                    tags_new.append((tag, prob))
                    continue

                has_mapped_a_tag = has_mapped_a_tag or tag not in mapped_tags

                for mapped_tag in mapped_tags:
                    existing_tag = next(filter(lambda t: t[0] == mapped_tag, tags_new), None)

                    if existing_tag is not None:
                        prob = max(prob, existing_tag[1])
                        tags_new.remove(existing_tag)

                    tags_new.append((mapped_tag, prob))
                    # print('tags_new.append((mapped_tag, prob))', mapped_tag, prob)

            tags_old = tags_new

            if not has_mapped_a_tag:
                break
        else:
            raise AssertionError('token mapping likely contains a recursion')

        # print('map_tokens', tags_new)
        return tags_new


    def _clean_ratings(items: List[Tuple[str, float]]):
        items_new = []
        for k, v in items:
            items_new.append([ k.removeprefix('rating_'), v])
        return items_new


    # print('character_res', len(character_res.items()), len(_replace_underscore(_threshold(character_res.items(), character_thresh, character_mcut_enabled))))
    # print('general_res', len(general_res.items()), len(_replace_underscore(_threshold(general_res.items(), general_thresh, general_mcut_enabled))))

    
    character_res = _replace_underscore(character_res.items())
    general_res = _replace_underscore(general_res.items())

    character_tags = set([ k for k, _ in character_res ])
    general_tags = set([ k for k, _ in general_res ])

    character_res = _threshold(character_res, character_thresh, character_mcut_enabled)
    general_res = _trim_general_tag_dupes(_threshold(general_res, general_thresh, general_mcut_enabled))

    tag_string = _generate_string(character_res, general_res)

    rating = _clean_ratings(rating.items())
    
    # recreate results to match the changes

    tag_res = list(character_res) + list(general_res)
    tag_res = sorted(tag_res, key=lambda x: x[1], reverse=True)
    tag_res = _map_tokens(tag_res)
    tag_res = _ban_tokens(tag_res)
    tag_res = sorted(tag_res, key=lambda x: x[1], reverse=True)

    character_res = [ (k, v) for k, v in tag_res if k in character_tags ]
    general_res = [ (k, v) for k, v in tag_res if k in general_tags ]

    return tag_string, dict(rating), dict(general_res), dict(character_res)

def post_process_manual_edits(
        initial_tags: str,
        edited_tags: str,
        new_tags: str,
):
    import difflib

    def merge_diff(initial, diff):
        updated = list(initial)
        diff = [ (i, tag_diff[:2], tag_diff[2:]) for i, tag_diff in enumerate(diff)]

        for i, op, tag in diff:
            if op == '+ ':
                # skip if already exists
                # FIXME: what if tags are duplicated?
                try:
                    updated.index(tag)
                    continue
                except ValueError:
                    pass

                # print('adding', i, tag)

                # find where tag should be inserted at
                tag_i = i
                for j in range(i-1, -1, -1):
                    # print('  try', j, x2[j][2:], x)
                    try:
                        tag_i = updated.index(diff[j][2])
                        # print('  found', xi)
                        break
                    except ValueError:
                        continue

                # insert tag in appropriate place
                tag_i = min(len(updated), tag_i)
                updated.insert(tag_i+1, tag)
            elif op == '- ':
                # print('removing', i, tag)
                try:
                    updated.remove(tag)
                except ValueError:
                    pass

        return updated
    
    initial_tags = [tag.strip() for tag in initial_tags.split(',')]
    edited_tags = [tag.strip() for tag in edited_tags.split(',')]
    new_tags = [tag.strip() for tag in new_tags.split(',')]

    diff = list(difflib.ndiff(initial_tags, edited_tags))

    return ', '.join(merge_diff(new_tags, diff))
