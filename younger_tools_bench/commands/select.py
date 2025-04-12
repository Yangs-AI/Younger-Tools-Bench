#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-17 17:28:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-12 17:00:59
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_tools_bench.commands import equip_logger


@click.group(name='select')
def select():
    pass

@select.command(name='basic')
@click.option('--allow-filepath',   required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), multiple=True, help='The directory where the data will be loaded.')
@click.option('--whole-dirpath',    required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--split-dirpath',    required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--output-dirpath',   required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--round',            required=False, type=int, default=10, help='The number of workers to use for processing.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_filter(
    allow_filepath,
    whole_dirpath,
    split_dirpath,
    output_dirpath,
    round,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_tools_bench.basic import select

    select.main(allow_filepath, whole_dirpath, split_dirpath, output_dirpath, round)
