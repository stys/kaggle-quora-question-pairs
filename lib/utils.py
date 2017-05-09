# -*- coding: utf-8 -*-

from os import makedirs as os_makedirs
import errno

from pyhocon.tool import HOCONConverter


def makedirs(path):
    try:
        os_makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def dump_config(conf, filename, output_format='hocon'):
    lines = HOCONConverter.convert(conf, output_format, indent=4)
    with open(filename, 'w') as fh:
        fh.writelines(lines)


def json_string_config(conf):
    lines = HOCONConverter.convert(conf, indent=1)
    return ' '.join(lines)