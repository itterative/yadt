from typing import Tuple, Dict, List

import numpy as np

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
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
]

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

        tags_new = []
        for tag, prob in tags:
            tags_new.append([ tag.replace('_', ' ') if tag not in kaomojis else tag, prob])
        return tags_new

    def _generate_string(character_res: List[Tuple[str, float]], general_res: List[Tuple[str, float]]):
        character_res = _trim_general_tag_dupes(character_res)
        general_res = list(map(lambda x: (x[0], x[1] - 1.0), general_res))

        # print('generate_string', character_res, general_res)

        sorted_tags = sorted(character_res + general_res, key=lambda x: x[1], reverse=True)
        sorted_tags = _map_tokens(sorted_tags)
        sorted_tags = _keep_tokens(sorted_tags)
        sorted_tags = _prefix_tokens(sorted_tags)
        sorted_tags = _ban_tokens(sorted_tags)
        sorted_tags = _replace_underscore(sorted_tags)
        sorted_tags = sorted(sorted_tags, key=lambda x: x[1], reverse=True)

        sorted_tags = list(map(lambda x: x[0], sorted_tags))
        generated_string = ', '.join(sorted_tags)

        if escape_brackets:
            generated_string = generated_string.replace("(", "\\(").replace(")", "\\)")

        return generated_string
    
    def _trim_general_tag_dupes(tags: List[Tuple[str, float]]):
        if not trim_general_tag_dupes:
            return tags

        def matches_tag(tag, s):
            tag = tag.split()
            s = s.split()
            len_tag = len(tag)
            return any(tag == s[i:len_tag+i] for i in range(len(s)-len_tag+1))
        
        tags_new = list(sorted(tags, key=lambda x: x[1], reverse=True))

        removed = True
        while removed:
            removed = False
            for i in range(len(tags_new)):
                tag = tags_new[i]
                found = next(filter(lambda s: s[0] != tag and matches_tag(tag[0], s[0]), tags_new), None)

                if found is None:
                    continue

                found = True
                tags_new.remove(tag)
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

    character_res = _replace_underscore(_threshold(character_res.items(), character_thresh, character_mcut_enabled))
    general_res = _replace_underscore(_threshold(general_res.items(), general_thresh, general_mcut_enabled))
    rating = _clean_ratings(rating.items())

    return _generate_string(character_res, general_res), dict(rating), dict(general_res), dict(character_res)
