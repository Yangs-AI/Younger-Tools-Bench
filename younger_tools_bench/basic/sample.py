#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-04-11 22:59:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-12 09:56:10
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import ast
import random
import pandas
import pathlib

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger_logics_ir.modules import LogicX


def get_ops(logicx_filepath: pathlib.Path) -> set[tuple[str, str]]:
    ops_set = set()
    logicx = LogicX()
    logicx.load(logicx_filepath)

    for node_index in logicx.operator_indices:
        node_tuid = logicx.node_tuid_feature(node_index)

        op_info = ast.literal_eval(node_tuid.split('-', 1)[1])
        op = (op_info[0], op_info[1])
        ops_set.add(op)
    return ops_set


def sample_skeleton(allow_filepath: pathlib.Path, whole_dirpath: pathlib.Path, split_dirpath: pathlib.Path, output_dirpath: pathlib.Path):
    logger.info(f' - Loading Allowed Ops ...')
    df = pandas.read_csv(allow_filepath, header=None)
    df.fillna("", inplace=True)
    allowed_ops: set[tuple[str, str]] = set(map(tuple, df.to_numpy()))

    whole_list: dict[str, set[tuple[str, str]]] = dict()
    split_list: dict[str, set[tuple[str, str]]] = dict()

    whole_uncover_ops: set[tuple[str, str]] = set(allowed_ops)
    whole_filepaths = [whole_filepath for whole_filepath in whole_dirpath.iterdir()]
    logger.info(f' - Check Whole Graphs (# {len(whole_filepaths)}) ...')
    for whole_filepath in whole_filepaths:
        ops_set = get_ops(whole_filepath)
        if ops_set and ops_set.issubset(allowed_ops):
            whole_list[whole_filepath.name] = list(ops_set)
            for op in ops_set:
                whole_uncover_ops.discard(op)
    logger.info(f' -  # {len(whole_uncover_ops)} Uncover Ops')

    save_json(whole_list, output_dirpath.joinpath('whole.json'))

    split_uncover_ops: set[tuple[str, str]] = set(allowed_ops)
    split_filepaths = [split_filepath for split_filepath in split_dirpath.iterdir()]
    random.shuffle(split_filepaths)
    for split_filepath in split_filepaths:
        if len(split_list) >= 10000 and len(split_uncover_ops) == 0:
            break
        ops_set = get_ops(split_filepath)
        if ops_set and ops_set.issubset(allowed_ops):
            split_list[split_filepath.name] = list(ops_set)
            for op in ops_set:
                split_uncover_ops.discard(op)
    logger.info(f' -  # {len(split_uncover_ops)} Uncover Ops')

    save_json(whole_list, output_dirpath.joinpath('whole.json'))


def main(allow_filepath: pathlib.Path, whole_dirpath: pathlib.Path, split_dirpath: pathlib.Path, output_dirpath: pathlib.Path):
    logger.info(f'... Sample Skeletons ...')
    sample_skeleton(allow_filepath, whole_dirpath, split_dirpath, output_dirpath)
