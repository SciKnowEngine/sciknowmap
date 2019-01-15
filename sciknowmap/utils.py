from __future__ import print_function

def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):                                                                                               
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-' + name[:1],
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('-n' + name[:1], '--no' + name, dest=name, action='store_false')
