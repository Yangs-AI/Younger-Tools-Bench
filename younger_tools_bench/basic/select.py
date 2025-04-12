#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-04-11 22:58:57
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-12 16:54:29
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import csv
import json
import numpy
import pandas
import random
import pathlib

from typing import Literal

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from younger.commons.io import load_json, create_dir
from younger.commons.logging import logger


class SkeletonSelector:
    def __init__(self, allowed_ops: set[tuple[str, str]], whole_hsh2ops: dict[str, set[tuple[str, str]]], whole_hsh2emb: dict[str, list[float]], split_hsh2ops: dict[str, set[tuple[str, str]]], split_hsh2emb: dict[str, list[float]]):
        self.allowed_ops = allowed_ops

        self.all_hashes = list(whole_hsh2ops.keys()) + list(split_hsh2ops.keys())
        self.all_ops = [whole_hsh2ops[h] for h in whole_hsh2ops] + [split_hsh2ops[h] for h in split_hsh2ops]
        self.all_emb = numpy.array([whole_hsh2emb[h] for h in whole_hsh2ops] + [split_hsh2emb[h] for h in split_hsh2ops])

        self.whole_hash_set = set(whole_hsh2ops.keys())
        self.split_hash_set = set(split_hsh2ops.keys())

    def _evaluate(self, selected):
        covered_ops = set()
        for i in selected:
            covered_ops |= self.all_ops[i]
        coverage = len(covered_ops & self.allowed_ops) / len(self.allowed_ops) if self.allowed_ops else 1.0

        if len(selected) > 1:
            emb = self.all_emb[selected]
            dist = pairwise_distances(emb)
            diversity = numpy.mean(numpy.partition(dist, 1)[:, 1])
        else:
            diversity = 0.0
        return coverage, diversity

    def select_phase_controlled(self, base_method: Literal['greedy'] = 'greedy', k: int = 15, round: int = 10, weight: int = 0.05, cluster: int = 30, top_k: int = 3, seed: int = 0):
        def run_method(method, indices):
            subset_ops = [self.all_ops[i] for i in indices]
            subset_emb = self.all_emb[indices]
            subset_selector = SkeletonSelector(self.allowed_ops, {}, {}, {}, {})
            subset_selector.all_hashes = [self.all_hashes[i] for i in indices]
            subset_selector.all_ops = subset_ops
            subset_selector.all_emb = subset_emb
            subset_selector.whole_hash_set = self.whole_hash_set & set(subset_selector.all_hashes)
            subset_selector.split_hash_set = self.split_hash_set & set(subset_selector.all_hashes)
            return getattr(subset_selector, f"select_{method}")(k, round, weight, cluster, top_k, seed)

        whole_indices = [i for i, h in enumerate(self.all_hashes) if h in self.whole_hash_set]
        selected, info, best_run = run_method(base_method, whole_indices)
        cov, _ = self._evaluate(selected)

        if cov < 1.0:
            remaining_ops = self.allowed_ops - set().union(*[self.all_ops[i] for i in selected])
            remaining_indices = [i for i, h in enumerate(self.all_hashes)
                                 if h in self.split_hash_set and h not in {self.all_hashes[i] for i in selected}]
            sub_selector = SkeletonSelector(remaining_ops, {}, {}, {}, {})
            sub_selector.all_hashes = [self.all_hashes[i] for i in remaining_indices]
            sub_selector.all_ops = [self.all_ops[i] for i in remaining_indices]
            sub_selector.all_emb = self.all_emb[remaining_indices]
            sub_selector.whole_hash_set = set()
            sub_selector.split_hash_set = set(sub_selector.all_hashes)
            sel_sub, _, _ = getattr(sub_selector, f"select_{base_method}")(k=k - len(selected), round=round, weight=weight, cluster=cluster, top_k=top_k, seed=seed)
            selected += [remaining_indices[i] for i in sel_sub]
        return selected, info, best_run

    def select_greedy(self, k: int = 15, round: int = 10, weight: int = 0.05, cluster: int = 30, top_k: int = 3, seed: int = 0):
        best_selection, best_cov, best_div, best_run = list(), -1, -1, None
        run_infos = list()

        for run in range(round):
            random.seed(seed + run)
            uncovered = set(self.allowed_ops)
            remaining = list(range(len(self.all_hashes)))
            selected = list()
            if not remaining:
                break
            first = random.choice(remaining)
            selected.append(first)
            uncovered -= self.all_ops[first]
            remaining.remove(first)
            while len(selected) < k and uncovered:
                best_idx, best_score = None, -1
                for i in remaining:
                    op_gain = len((self.all_ops[i] & uncovered))
                    emb_dist = pairwise_distances(self.all_emb[i:i+1], self.all_emb[selected]).min()
                    score = op_gain + weight * emb_dist
                    if score > best_score:
                        best_score, best_idx = score, i
                if best_idx is None:
                    break
                selected.append(best_idx)
                uncovered -= self.all_ops[best_idx]
                remaining.remove(best_idx)
            coverage, diversity = self._evaluate(selected)
            run_infos.append({"run": run, "coverage": coverage, "diversity": diversity, "count": len(selected)})
            logger.info(f'run: {run} | coverage: {coverage} | diversity: {diversity}, count: {len(selected)}')
            if coverage > best_cov or (abs(coverage - best_cov) < 1e-5 and diversity > best_div):
                best_cov, best_div, best_selection, best_run = coverage, diversity, selected[:], run
        return best_selection, run_infos, best_run

    def select_across(self, k: int = 15, round: int = 10, weight: int = 0.05, cluster: int = 30, top_k: int = 3, seed: int = 0):
        best_selection, best_cov, best_div, best_run = list(), -1, -1, None
        run_infos = list()

        for run in range(round):
            random.seed(seed + run)
            kmeans = KMeans(n_clusters=cluster, random_state=seed + run, n_init=1)
            labels = kmeans.fit_predict(self.all_emb)
            centers = kmeans.cluster_centers_
            candidate_pool = list()
            for cid in range(cluster):
                members = [i for i, lbl in enumerate(labels) if lbl == cid]
                if not members:
                    continue
                scored = list()
                for i in members:
                    op_score = len(self.all_ops[i] & self.allowed_ops)
                    dist = numpy.linalg.norm(self.all_emb[i] - centers[cid])
                    score = op_score + weight * dist
                    scored.append((score, i))
                top_indices = [i for _, i in sorted(scored, reverse=True)[:top_k]]
                candidate_pool.extend(top_indices)
            candidate_pool = candidate_pool[:k]
            coverage, diversity = self._evaluate(candidate_pool)
            run_infos.append({"run": run, "coverage": coverage, "diversity": diversity, "count": len(candidate_pool)})
            logger.info(f'run: {run} | coverage: {coverage} | diversity: {diversity}, count: {len(candidate_pool)}')
            if coverage > best_cov or (abs(coverage - best_cov) < 1e-5 and diversity > best_div):
                best_cov, best_div, best_selection, best_run = coverage, diversity, candidate_pool[:], run
        return best_selection, run_infos, best_run

    def select_hybrid(self, k: int = 15, round: int = 10, weight: int = 0.05, cluster: int = 30, top_k: int = 3, seed: int = 0):
        best_selection, best_cov, best_div, best_run = list(), -1, -1, None
        run_infos = list()
        total = len(self.all_hashes)

        for run in range(round):
            random.seed(seed + run)
            kmeans = KMeans(n_clusters=cluster, random_state=seed + run, n_init=1)
            labels = kmeans.fit_predict(self.all_emb)
            centers = kmeans.cluster_centers_
            pool = list()
            for cid in range(cluster):
                members = [i for i, lbl in enumerate(labels) if lbl == cid]
                if not members:
                    continue
                scored = list()
                for i in members:
                    op_score = len(self.all_ops[i] & self.allowed_ops)
                    dist = numpy.linalg.norm(self.all_emb[i] - centers[cid])
                    score = op_score + weight * dist
                    scored.append((score, i))
                top_indices = [i for _, i in sorted(scored, reverse=True)[:top_k]]
                pool.extend(top_indices)
            pool = list(set(pool))
            uncovered = set(self.allowed_ops)
            selected = list()
            remaining = pool[:]
            if remaining:
                first = random.choice(remaining)
                selected.append(first)
                uncovered -= self.all_ops[first]
                remaining.remove(first)
            while len(selected) < k and uncovered:
                best_idx, best_score = None, -1
                for i in remaining:
                    op_gain = len((self.all_ops[i] & uncovered))
                    emb_dist = pairwise_distances(self.all_emb[i:i+1], self.all_emb[selected]).min()
                    score = op_gain + weight * emb_dist
                    if score > best_score:
                        best_score, best_idx = score, i
                if best_idx is None:
                    break
                selected.append(best_idx)
                uncovered -= self.all_ops[best_idx]
                remaining.remove(best_idx)
            coverage, diversity = self._evaluate(selected)
            run_infos.append({"run": run, "coverage": coverage, "diversity": diversity, "count": len(selected)})
            logger.info(f'run: {run} | coverage: {coverage} | diversity: {diversity}, count: {len(selected)}')
            if coverage > best_cov or (abs(coverage - best_cov) < 1e-9 and diversity > best_div):
                best_cov, best_div, best_selection, best_run = coverage, diversity, selected[:], run
        return best_selection, run_infos, best_run

    def save_results(self, method_name, selected_indices, runs_info, best_run, output_dirpath):
        selected_path = output_dirpath / f"{method_name}_selected.csv"
        runs_path = output_dirpath / f"{method_name}_runs.csv"
        best_path = output_dirpath / f"{method_name}_best.json"

        with open(selected_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["graph_hash", "is_full"])
            for i in selected_indices:
                is_full = self.all_hashes[i] in self.whole_hash_set
                writer.writerow([self.all_hashes[i], str(is_full).upper()])

        with open(runs_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["run", "coverage", "diversity", "count"])
            writer.writeheader()
            for info in runs_info:
                writer.writerow(info)

        best_info = next((info for info in runs_info if info["run"] == best_run), None)
        selected_hashes = [self.all_hashes[i] for i in selected_indices]
        best_json = {
            "method": method_name,
            "best_run": best_run,
            "coverage": best_info["coverage"] if best_info else None,
            "diversity": best_info["diversity"] if best_info else None,
            "selected_count": len(selected_indices),
            "selected_hashes": selected_hashes
        }
        with open(best_path, 'w', encoding='utf-8') as f:
            json.dump(best_json, f, ensure_ascii=False, indent=2)


def main(
    allow_filepath: pathlib.Path,
    whole_dirpath: pathlib.Path,
    split_dirpath: pathlib.Path,
    output_dirpath: pathlib.Path,
    round: int,
):
    logger.info(f'... Select Skeletons ...')

    logger.info(f' - Loading Allowed Ops ...')
    df = pandas.read_csv(allow_filepath, header=None)
    df.fillna("", inplace=True)
    allowed_ops: set[tuple[str, str]] = set(map(tuple, df.to_numpy()))

    logger.info(f' - Loading Whole Graph Embeddings & OPSETs ...')
    whole_hsh2ops: dict[str, list[tuple[str, str]]] = load_json(whole_dirpath.joinpath('whole.json')) # {hash: [(op_type, domain)]}
    whole_hsh2ops: dict[str, set[tuple[str, str]]] = {whole_hsh: set(whole_ops) for whole_hsh, whole_ops in whole_hsh2ops.items()}
    hsh_df = pandas.read_csv(whole_dirpath.joinpath('graph_hashes.csv'))
    emb_df = pandas.read_csv(whole_dirpath.joinpath('graph_embeddings.csv'))
    whole_hashes = hsh_df[0].astype(str).tolist()
    whole_embeddings = emb_df.to_numpy().tolist()
    assert len(whole_hashes) == len(whole_embeddings)
    whole_hsh2emb: dict[str, list[float]] = dict(zip(whole_hashes, whole_embeddings))

    logger.info(f' - Loading Split Graph Embeddings & OPSETs ...')
    split_hsh2ops: dict[str, list[tuple[str, str]]] = load_json(split_dirpath.joinpath('split.json')) # {hash: [(op_type, domain)]}
    split_hsh2ops: dict[str, set[tuple[str, str]]] = {split_hsh: set(split_ops) for split_hsh, split_ops in split_hsh2ops.items()}
    hsh_df = pandas.read_csv(split_dirpath.joinpath('graph_hashes.csv'))
    emb_df = pandas.read_csv(split_dirpath.joinpath('graph_embeddings.csv'))
    split_hashes = hsh_df[0].astype(str).tolist()
    split_embeddings = emb_df.to_numpy().tolist()
    assert len(split_hashes) == len(split_embeddings)
    split_hsh2emb: dict[str, list[float]] = dict(zip(split_hashes, split_embeddings))

    create_dir(output_dirpath)
    selector = SkeletonSelector(allowed_ops, whole_hsh2ops, whole_hsh2emb, split_hsh2ops, split_hsh2emb)

    all_selected_indices, all_informations, best_round = selector.select_greedy(round=round)
    selector.save_results("greedy", all_selected_indices, all_informations, best_round, output_dirpath)

    all_selected_indices, all_informations, best_round = selector.select_across(round=round)
    selector.save_results("across", all_selected_indices, all_informations, best_round, output_dirpath)

    all_selected_indices, all_informations, best_round = selector.select_hybrid(round=round)
    selector.save_results("hybrid", all_selected_indices, all_informations, best_round, output_dirpath)

    all_selected_indices, all_informations, best_round = selector.select_phase_controlled('greedy', round=round)
    selector.save_results("phase_greedy", all_selected_indices, all_informations, best_round, output_dirpath)

    all_selected_indices, all_informations, best_round = selector.select_phase_controlled('greedy', round=round)
    selector.save_results("phase_across", all_selected_indices, all_informations, best_round, output_dirpath)

    all_selected_indices, all_informations, best_round = selector.select_phase_controlled('greedy', round=round)
    selector.save_results("phase_hybrid", all_selected_indices, all_informations, best_round, output_dirpath)
