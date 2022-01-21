"""
handle config update as EasyDict
"""
from easydict import EasyDict
from pprint import pprint
import yaml

def is_number_or_bool_or_none(x):
    try:
        float(x)
        return True
    except ValueError:
        return x in ['True', 'False', 'None']


def add_quotation_to_string(s, split_chars=None):

    if split_chars is None:
        split_chars = ['[', ']', '{', '}', ',', ' ']
        if '{' in s and '}' in s:
            split_chars.append(':')
    s_mark, marker = s, chr(1)
    for split_char in split_chars:
        s_mark = s_mark.replace(split_char, marker)

    s_quoted = ''
    for value in s_mark.split(marker):
        if len(value) == 0:
            continue
        st = s.find(value)
        if is_number_or_bool_or_none(value):
            s_quoted += s[:st] + value
        elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
            s_quoted += s[:st] + value
        else:
            s_quoted += s[:st] + '"' + value + '"'
        s = s[st + len(value):]

    return s_quoted + s

def update_config(cfg, cfg_argv, delimiter='='):
    r""" Update cfg with list from argparser

    Args:
        cfg (easy dict): the cfg to be updated by the argv
        cfg_argv: the new config list, like ['epoch=10', 'save.last=False']
        dilimeter: the dilimeter between key and value of the given config
    """

    def resolve_cfg_with_legality_check(keys):
        r""" Resolve the parent and leaf from given keys and check their legality.

        Args:
            keys: The hierarchical keys of global cfg

        Returns:
            the resolved parent adict obj and its legal key to be upated.
        """

        obj, obj_repr = cfg, 'cfg'
        for idx, sub_key in enumerate(keys):
            if not isinstance(obj, EasyDict) or sub_key not in obj:
                raise ValueError(f'Undefined attribute "{sub_key}" detected for "{obj_repr}"')
            if idx < len(keys) - 1:
                obj = obj.get(sub_key)
                obj_repr += f'.{sub_key}'
        return obj, sub_key

    # Exit if no cfg_argv is given, the cfg will not be force converted to adict()
    if len(cfg_argv) == 0:
        return

    # Update all dict to adict so that user-defined param could be partially updated with argv

    for str_argv in cfg_argv:
        item = str_argv.split(delimiter, 1)
        assert len(item) == 2, "Error argv (must be key=value): " + str_argv
        key, value = item
        obj, leaf = resolve_cfg_with_legality_check(key.split('.'))
        obj[leaf] = eval(add_quotation_to_string(value))


def print_cfg(config):
    """
    print out an easy dict iteratively
    """
    pprint(config)

def dump_config(config, save_path):
    """
    save an easy dict config as a .yaml file to save path.
    """
    with open(save_path, "w+") as f:
        yaml.dump(config, f, default_flow_style=False)