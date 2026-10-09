"""Load YAML launch options while retaining argparse validation and CLI overrides."""

import argparse
import copy
from pathlib import Path
import sys

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently keeping the last value."""


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError('configuration keys must be strings')
        if key in mapping:
            raise ValueError(f'duplicate configuration key: {key}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping,
)


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None):
    """Parse flat YAML options, with explicit CLI arguments taking precedence.

    YAML keys use argparse destination names (``batch_size``); hyphenated option
    names are also accepted. Values pass through the original argparse types,
    choices, required options and mutually exclusive groups. YAML Path values
    are relative to the YAML file; CLI paths retain their usual cwd semantics.
    """
    parser.add_argument('--config', type=Path, help='YAML configuration file; CLI options override its values')
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        return parser.parse_args(argv)

    # Probe only explicit arguments, without enforcing requirements that YAML
    # may supply. Suppressed defaults distinguish omission from an explicit False.
    probe = copy.deepcopy(parser)
    probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    for group in probe._mutually_exclusive_groups:
        group.required = False
    explicit, _ = probe.parse_known_args(argv)
    explicit = vars(explicit)
    config_path = explicit.get('config')
    if config_path is None:
        return parser.parse_args(argv)

    try:
        with config_path.open(encoding='utf-8') as stream:
            values = yaml.load(stream, Loader=_UniqueKeyLoader)
        if not isinstance(values, dict):
            raise ValueError('configuration must be a YAML mapping of option names to values')
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f'cannot load --config {config_path}: {exc}')

    actions = {}
    for action in parser._actions:
        if action.dest in ('help', 'config'):
            continue
        actions[action.dest] = action
        for option in action.option_strings:
            if option.startswith('--') and not option.startswith('--no-'):
                actions[option[2:].replace('-', '_')] = action

    normalized = {}
    for key, value in values.items():
        action = actions.get(key.replace('-', '_'))
        if action is None:
            parser.error(f'{config_path}: unknown configuration option {key!r}')
        if action.dest in normalized:
            parser.error(f'{config_path}: duplicate option {action.dest!r}')
        normalized[action.dest] = (action, value)

    overridden = set(explicit)
    grouped = set()
    for group in parser._mutually_exclusive_groups:
        members = {action.dest for action in group._group_actions}
        grouped.update(members)
        if members & overridden:
            overridden.update(members)

    config_argv = []
    for dest, (action, value) in normalized.items():
        if dest in overridden:
            continue
        # Null means unspecified, e.g. a checkpoint or EMA decay still to fill in.
        if value is None:
            if action.default is not None:
                parser.error(f'{config_path}: {dest} cannot be null')
            continue
        option = action.option_strings[0]
        if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
            if not isinstance(value, bool):
                parser.error(f'{config_path}: {dest} must be a YAML boolean (true or false)')
            # A false group member is not a selection (e.g. random_init: false).
            if value or (dest not in grouped and isinstance(action, argparse.BooleanOptionalAction)):
                config_argv.append(option if value else '--no-' + option[2:])
            continue

        is_list = action.nargs in ('+', '*')
        if is_list != isinstance(value, list):
            parser.error(f'{config_path}: {dest} must be a {"list" if is_list else "scalar"}')
        items = value if is_list else [value]
        if any(isinstance(item, bool) or not isinstance(item, (str, int, float)) for item in items):
            parser.error(f'{config_path}: {dest} must contain only string or numeric values')
        if action.type is Path:
            if not isinstance(value, str):
                parser.error(f'{config_path}: {dest} must be a path string')
            path = Path(value).expanduser()
            items = [str(path if path.is_absolute() else (config_path.parent / path).resolve())]
        if is_list:
            config_argv.extend([option, *(str(item) for item in items)])
        else:
            config_argv.append(f'{option}={items[0]}')

    return parser.parse_args(config_argv + argv)
