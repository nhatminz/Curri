from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .checkpoint import (
    capture_rng_state,
    load_training_state,
    resolve_resume_path,
    save_checkpoint,
)
from .config import RACConfig, save_config
from .critical_states import select_critical_states
from .curriculum import CurriculumState, PromptScore
from .data import load_records, render_prompt, tokenize_prompt
from .distributed import (
    DistributedContext,
    barrier,
    broadcast_indices,
    ddp_local_loss_scale,
    gather_objects,
    reduce_sum,
)
from .evaluator import should_evaluate
from .logging_utils import RunLogger
from .models import load_models
from .opd_loss import sequence_opd_loss
from .plotting import generate_all_plots, save_and_plot_prompt_usage
from .recoverability import aggregate_prompt, state_recoverability
from .rollout import Rollout, RolloutRequest, create_rollout_backend
from .scoring import DiagnosticScorer, choose_branch_candidates


LOGGER = logging.getLogger(__name__)


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return (
        float(np.nanmean(array))
        if array.size and np.any(np.isfinite(array))
        else float("nan")
    )


class RACOPDTrainer:
    def __init__(self, config: RACConfig, context: DistributedContext):
        self.config = config
        self.context = context
        if context.world_size > config.training.global_batch_size:
            raise ValueError("world_size cannot exceed the exact global batch size")
        self.output_dir = Path(config.paths.output_root) / config.training.run_name
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.resume_path = resolve_resume_path(
            config.checkpoint.resume, self.checkpoint_root
        )

        random.seed(config.training.seed + context.rank)
        np.random.seed(config.training.seed + context.rank)
        torch.manual_seed(config.training.seed + context.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed + context.rank)
            torch.backends.cuda.matmul.allow_tf32 = config.training.tf32
            torch.backends.cudnn.allow_tf32 = config.training.tf32

        self.records, self.schema = load_records(
            config.paths.train_data,
            split=config.data.train_split,
            prompt_field=config.data.prompt_field,
            answer_field=config.data.answer_field,
            id_field=config.data.id_field,
        )
        if config.training.dry_run:
            self.records = self.records[: min(8, len(self.records))]
            config.training.max_steps = 1
            config.training.global_batch_size = min(
                config.training.global_batch_size, len(self.records)
            )
            config.rollout.max_new_tokens = min(config.rollout.max_new_tokens, 32)
            config.evaluation.enabled = False
            LOGGER.warning(
                "Dry run: using %d prompts, one step, <=32 generated tokens, evaluation off",
                len(self.records),
            )
        self.prompt_ids = [record.prompt_id for record in self.records]
        c = config.curriculum
        self.curriculum = CurriculumState(
            self.prompt_ids,
            initial_score=c.initial_score,
            age_tau_steps=c.age_tau_steps,
            enable_staleness_decay=c.enable_staleness_decay,
            eps_explore=c.eps_explore,
            temperature=c.temperature,
            ema_beta=c.ema_beta,
            seed=c.seed,
        )

        student_path = str(self.resume_path / "student") if self.resume_path else None
        bundle = load_models(config, context.device, student_path)
        self.tokenizer = bundle.tokenizer
        self.teacher = bundle.teacher
        if context.world_size > 1:
            self.student: torch.nn.Module = DistributedDataParallel(
                bundle.student,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.student = bundle.student
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.start_step = 0
        if self.resume_path:
            state = load_training_state(
                self.resume_path,
                self.optimizer,
                self.scheduler,
                self.curriculum,
                context.rank,
            )
            self.start_step = int(state["step"])
            if len(state["rng_states_by_rank"]) != context.world_size:
                raise ValueError(
                    "Exact resume requires the same world size as the checkpoint"
                )
            LOGGER.info(
                "Resumed exact training state at step %d; next step is %d",
                self.start_step,
                self.start_step + 1,
            )

        self.rollout_backend = create_rollout_backend(
            config.rollout.backend,
            self.student,
            self.tokenizer,
            config.rollout,
            context.device,
        )
        self.scorer = DiagnosticScorer(
            self.student,
            self.teacher,
            context.device,
            stats_top_k=config.critical.stats_top_k,
            position_chunk_size=config.critical.score_position_chunk,
            branch_microbatch_size=config.critical.branch_microbatch_size,
        )
        self.run_logger = (
            RunLogger(self.output_dir, config.logging.tensorboard)
            if context.is_main
            else None
        )
        self.max_steps = self._resolve_max_steps()
        self.config.training.max_steps = self.max_steps
        if context.is_main:
            save_config(config, self.output_dir / "config_resolved.yaml")
        self.tokenized_prompts = self._prepare_prompts()
        self.data_metadata = {
            "train_path": config.paths.train_data,
            "dataset_size": len(self.records),
            "prompt_ids": self.prompt_ids,
            "schema": asdict(self.schema),
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        kwargs: dict[str, Any] = {
            "lr": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
        }
        if self.config.training.fused_adamw and self.context.device.type == "cuda":
            kwargs["fused"] = True
        try:
            return torch.optim.AdamW(self.student.parameters(), **kwargs)
        except (TypeError, RuntimeError) as exc:
            kwargs.pop("fused", None)
            LOGGER.warning("Fused AdamW unavailable (%s); using standard AdamW", exc)
            return torch.optim.AdamW(self.student.parameters(), **kwargs)

    def _build_scheduler(self) -> Any:
        if self.config.training.scheduler != "constant":
            raise ValueError(
                "Only the research-default constant scheduler is currently supported"
            )
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    def _resolve_max_steps(self) -> int:
        if self.config.training.max_steps > 0:
            resolved = self.config.training.max_steps
        else:
            resolved = (
                math.ceil(len(self.records) / self.config.training.global_batch_size)
                + self.config.training.extra_steps
            )
        draws = resolved * self.config.training.global_batch_size / len(self.records)
        LOGGER.info(
            "Resolved max_steps=%d; equivalent draws per dataset item=%.4f",
            resolved,
            draws,
        )
        if self.start_step > resolved:
            raise ValueError(
                f"Checkpoint step {self.start_step} exceeds max_steps {resolved}"
            )
        return resolved

    def _prepare_prompts(self) -> list[tuple[int, ...]]:
        prepared: list[tuple[int, ...]] = []
        iterator = tqdm(
            self.records,
            desc="Tokenizing prompt pool",
            disable=not self.context.is_main,
        )
        for record in iterator:
            rendered = render_prompt(
                self.tokenizer,
                record.prompt,
                self.config.data.prompt_template_mode,
                self.config.data.enable_thinking,
            )
            tokens = tokenize_prompt(
                self.tokenizer, rendered, self.config.data.max_prompt_tokens
            )
            prepared.append(tuple(tokens))
        return prepared

    def _sample_requests(
        self, step: int
    ) -> tuple[list[RolloutRequest], dict[str, float]]:
        if self.context.is_main:
            sampled, probabilities = self.curriculum.sample(
                step, self.config.training.global_batch_size
            )
            shared = torch.from_numpy(sampled)
            probability_metrics = {
                "sampling_probability_min": float(probabilities.min()),
                "sampling_probability_max": float(probabilities.max()),
                "sampling_entropy": float(
                    -np.sum(probabilities * np.log(probabilities))
                ),
                "sampling_effective_size": float(1.0 / np.sum(probabilities**2)),
            }
        else:
            shared = None
            probability_metrics = {}
        local_indices = broadcast_indices(
            shared, self.config.training.global_batch_size, self.context
        ).tolist()
        requests = [
            RolloutRequest(
                prompt_index=index,
                prompt_id=self.records[index].prompt_id,
                prompt_ids=self.tokenized_prompts[index],
            )
            for index in local_indices
        ]
        return requests, probability_metrics

    def _diagnose_rollouts(
        self, rollouts: list[Rollout], step: int
    ) -> tuple[list[PromptScore], list[dict[str, Any]], dict[str, Any]]:
        config = self.config.critical
        special_ids = set(int(item) for item in self.tokenizer.all_special_ids)
        eos_value = self.tokenizer.eos_token_id
        eos_ids = (
            set(eos_value if isinstance(eos_value, list) else [eos_value])
            if eos_value is not None
            else set()
        )
        invalid_control_ids = special_ids - eos_ids
        prompt_results: list[PromptScore] = []
        critical_rows: list[dict[str, Any]] = []
        scoring_seconds = 0.0
        selection_seconds = 0.0
        branch_seconds = 0.0
        branch_probes = 0
        eos_candidates = 0
        cache_reused = 0

        for rollout in rollouts:
            _sync_cuda(self.context.device)
            started = time.perf_counter()
            statistics = self.scorer.score(rollout)
            _sync_cuda(self.context.device)
            scoring_seconds += time.perf_counter() - started

            started = time.perf_counter()
            selected = select_critical_states(
                statistics.dplus,
                statistics.compatibility,
                target=config.target,
                num_segments=config.num_segments,
                global_peaks=config.global_peaks,
                change_points=config.change_points,
                change_lag=config.change_lag,
                min_gap_tokens=config.min_gap_tokens,
            )
            selection_seconds += time.perf_counter() - started
            state_values = []
            for critical in selected:
                position = critical.position
                alpha_all = statistics.topk_positive_correction[position]
                candidate_ids, candidate_alpha, eos_count = choose_branch_candidates(
                    statistics.topk_ids[position],
                    statistics.topk_student_prob[position],
                    statistics.topk_teacher_prob[position],
                    config.branch_candidates,
                    invalid_control_ids,
                    eos_ids,
                )
                eos_candidates += eos_count
                _sync_cuda(self.context.device)
                branch_started = time.perf_counter()
                if candidate_ids.size:
                    prefix = rollout.prompt_ids + rollout.response_ids[:position]
                    next_compatibility, used_cache = (
                        self.scorer.probe_next_compatibility(
                            prefix, candidate_ids.tolist()
                        )
                    )
                    branch_probes += int(candidate_ids.size)
                    cache_reused += int(used_cache)
                else:
                    next_compatibility = np.empty((0,), dtype=np.float32)
                _sync_cuda(self.context.device)
                branch_seconds += time.perf_counter() - branch_started
                state = state_recoverability(
                    float(statistics.dplus[position]),
                    float(alpha_all.sum()),
                    candidate_alpha,
                    next_compatibility,
                    config.eps_num,
                )
                state_values.append(state)
                response_length = len(rollout.response_ids)
                critical_rows.append(
                    {
                        "step": step,
                        "prompt_id": rollout.prompt_id,
                        "prompt_index": rollout.prompt_index,
                        "absolute_position": position,
                        "response_length": response_length,
                        "normalized_position": position / response_length
                        if response_length
                        else float("nan"),
                        "selection_reason": "|".join(critical.reasons),
                        "Dplus_t": state.dplus,
                        "C_t": float(statistics.compatibility[position]),
                        "A_t": state.accessibility,
                        "F_t": state.future_compatibility,
                        "B_t": state.bridgeability,
                        "valid": state.valid,
                        "branch_count": int(candidate_ids.size),
                    }
                )
            prompt = aggregate_prompt(state_values)
            prompt_results.append(
                PromptScore(
                    prompt_index=rollout.prompt_index,
                    prompt_id=rollout.prompt_id,
                    teachability=prompt.teachability,
                    need=prompt.need,
                    recoverability=prompt.recoverability,
                    rollout_length=len(rollout.response_ids),
                    valid_critical_states=prompt.valid_states,
                )
            )
            del statistics
        stats = {
            "diagnostic_scoring_seconds": scoring_seconds,
            "critical_selection_seconds": selection_seconds,
            "branch_probe_seconds": branch_seconds,
            "branch_probes": branch_probes,
            "eos_branch_candidates_skipped": eos_candidates,
            "critical_prefix_cache_states": cache_reused,
        }
        return prompt_results, critical_rows, stats

    def _train_update(
        self, rollouts: list[Rollout]
    ) -> tuple[float, float, float, float]:
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        scale = ddp_local_loss_scale(
            self.context.world_size, self.config.training.global_batch_size
        )
        local_loss_sum = 0.0
        _sync_cuda(self.context.device)
        started = time.perf_counter()
        for index, rollout in enumerate(rollouts):
            should_sync = index == len(rollouts) - 1
            sync_context = (
                contextlib.nullcontext()
                if should_sync or not hasattr(self.student, "no_sync")
                else self.student.no_sync()
            )
            with sync_context:
                sequence_loss = sequence_opd_loss(
                    self.student,
                    self.teacher,
                    rollout,
                    self.config.training.opd_top_k,
                    self.context.device,
                )
                local_loss_sum += float(sequence_loss.detach())
                (sequence_loss * scale).backward()
                del sequence_loss
        _sync_cuda(self.context.device)
        train_seconds = time.perf_counter() - started
        global_loss = (
            reduce_sum(local_loss_sum, self.context)
            / self.config.training.global_batch_size
        )
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), self.config.training.max_grad_norm
        )
        grad_norm = float(grad_norm_tensor.detach())
        _sync_cuda(self.context.device)
        optimizer_started = time.perf_counter()
        self.optimizer.step()
        self.scheduler.step()
        _sync_cuda(self.context.device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        return global_loss, grad_norm, train_seconds, optimizer_seconds

    def _save(self, step: int) -> Path:
        local_rng = capture_rng_state()
        gathered_rng = gather_objects(local_rng, self.context)
        checkpoint_path = self.checkpoint_root / f"step_{step:06d}"
        if self.context.is_main:
            assert gathered_rng is not None
            checkpoint_path = save_checkpoint(
                self.checkpoint_root,
                step,
                self.student,
                self.tokenizer,
                self.optimizer,
                self.scheduler,
                self.curriculum,
                gathered_rng,
                self.data_metadata,
                self.config,
                self.config.checkpoint.keep_last_n,
            )
            LOGGER.info("Saved checkpoint %s", checkpoint_path)
        barrier(self.context)
        return checkpoint_path

    def _run_evaluation(self, model_path: str, step: int) -> float:
        if not self.config.evaluation.enabled:
            return 0.0
        barrier(self.context)
        local_rng = capture_rng_state()
        started = time.perf_counter()
        if self.context.is_main:
            # A fresh process gives each vLLM evaluation a clean CUDA lifecycle and
            # releases its engine memory deterministically before training continues.
            project_root = Path(__file__).resolve().parent.parent
            environment = os.environ.copy()
            visible = environment.get("CUDA_VISIBLE_DEVICES", "")
            environment["CUDA_VISIBLE_DEVICES"] = (
                visible.split(",")[0] if visible else "0"
            )
            for key in list(environment):
                if key in {
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "LOCAL_WORLD_SIZE",
                    "GROUP_RANK",
                    "ROLE_RANK",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                } or key.startswith("TORCHELASTIC_"):
                    environment.pop(key, None)
            command = [
                sys.executable,
                str(project_root / "evaluate.py"),
                "--config",
                str(self.output_dir / "config_resolved.yaml"),
                "--model",
                model_path,
                "--step",
                str(step),
                "--output_dir",
                str(self.output_dir),
                "--set",
                "evaluation.tensor_parallel_size=1",
                "--set",
                "evaluation.data_parallel_size=1",
            ]
            subprocess.run(command, cwd=project_root, env=environment, check=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        barrier(self.context)
        # Evaluation is observational and must not perturb the subsequent training RNG stream.
        from .checkpoint import restore_rng_state

        restore_rng_state(local_rng)
        return time.perf_counter() - started

    def train(self) -> None:
        if self.start_step == 0 and should_evaluate(
            0, self.max_steps, self.config.evaluation.every_steps
        ):
            self._run_evaluation(self.config.paths.student_model, 0)
        progress = tqdm(
            range(self.start_step + 1, self.max_steps + 1),
            initial=self.start_step,
            total=self.max_steps,
            desc="RAC-OPD",
            disable=not self.context.is_main,
        )
        for step in progress:
            step_started = time.perf_counter()
            if self.context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.context.device)

            started = time.perf_counter()
            requests, probability_metrics = self._sample_requests(step)
            sampling_seconds = time.perf_counter() - started

            self.rollout_backend.sync_weights(step - 1)
            _sync_cuda(self.context.device)
            started = time.perf_counter()
            rollouts = self.rollout_backend.generate(requests)
            _sync_cuda(self.context.device)
            rollout_seconds = time.perf_counter() - started
            original_rollout_tokens = [rollout.response_ids for rollout in rollouts]

            local_results, local_critical, diagnostic_stats = self._diagnose_rollouts(
                rollouts, step
            )
            if [
                rollout.response_ids for rollout in rollouts
            ] != original_rollout_tokens:
                raise RuntimeError(
                    "Counterfactual diagnostics mutated the original rollout"
                )
            gathered_results = gather_objects(local_results, self.context)
            gathered_critical = gather_objects(local_critical, self.context)
            gathered_diagnostic_stats = gather_objects(diagnostic_stats, self.context)
            if self.context.is_main:
                assert gathered_results is not None and gathered_critical is not None
                flat_results = [
                    item for rank_items in gathered_results for item in rank_items
                ]
                flat_critical = [
                    item for rank_items in gathered_critical for item in rank_items
                ]
                for result in flat_results:
                    self.curriculum.update(result, step)
                assert self.run_logger is not None
                self.run_logger.log_critical_states(flat_critical)
            else:
                flat_results = []
                flat_critical = []

            if self.context.device.type == "cuda":
                torch.cuda.empty_cache()
            global_loss, grad_norm, train_seconds, optimizer_seconds = (
                self._train_update(rollouts)
            )

            eval_due = should_evaluate(
                step, self.max_steps, self.config.evaluation.every_steps
            )
            save_due = (
                step == self.max_steps
                or eval_due
                or self.config.checkpoint.save_every_steps > 0
                and step % self.config.checkpoint.save_every_steps == 0
            )
            checkpoint_path = self._save(step) if save_due else None
            evaluation_seconds = 0.0
            if eval_due and self.config.evaluation.enabled:
                assert checkpoint_path is not None
                evaluation_seconds = self._run_evaluation(
                    str(checkpoint_path / "student"), step
                )

            tokens_generated = sum(len(rollout.response_ids) for rollout in rollouts)
            local_lengths = [len(rollout.response_ids) for rollout in rollouts]
            local_summary = {
                "tokens": tokens_generated,
                "lengths": local_lengths,
                "clipped": sum(int(rollout.clipped) for rollout in rollouts),
                "rollout_seconds": rollout_seconds,
                "sampling_seconds": sampling_seconds,
                "train_seconds": train_seconds,
                "optimizer_seconds": optimizer_seconds,
                "peak_allocated_gb": torch.cuda.max_memory_allocated(
                    self.context.device
                )
                / 2**30
                if self.context.device.type == "cuda"
                else 0.0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(self.context.device)
                / 2**30
                if self.context.device.type == "cuda"
                else 0.0,
            }
            gathered_summary = gather_objects(local_summary, self.context)
            if self.context.is_main:
                assert (
                    gathered_summary is not None
                    and gathered_diagnostic_stats is not None
                )
                all_lengths = [
                    value
                    for summary in gathered_summary
                    for value in summary["lengths"]
                ]
                total_tokens = sum(summary["tokens"] for summary in gathered_summary)
                max_rollout_time = max(
                    summary["rollout_seconds"] for summary in gathered_summary
                )
                dplus_values = [row["Dplus_t"] for row in flat_critical]
                a_values = [row["A_t"] for row in flat_critical]
                f_values = [row["F_t"] for row in flat_critical if row["valid"]]
                b_values = [row["B_t"] for row in flat_critical if row["valid"]]
                metrics: dict[str, Any] = {
                    "step": step,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "opd_loss": global_loss,
                    "grad_norm": grad_norm,
                    "rollout_length_mean": _mean(all_lengths),
                    "rollout_length_min": min(all_lengths),
                    "rollout_length_max": max(all_lengths),
                    "rollout_clip_ratio": sum(
                        summary["clipped"] for summary in gathered_summary
                    )
                    / len(all_lengths),
                    "mean_Dplus": _mean(dplus_values),
                    "mean_A": _mean(a_values),
                    "mean_F": _mean(f_values),
                    "mean_B": _mean(b_values),
                    "mean_prompt_G": _mean(item.need for item in flat_results),
                    "mean_prompt_R": _mean(
                        item.recoverability for item in flat_results
                    ),
                    "mean_prompt_T": _mean(item.teachability for item in flat_results),
                    **probability_metrics,
                    "valid_critical_states": sum(
                        item.valid_critical_states for item in flat_results
                    ),
                    "selected_critical_states": len(flat_critical),
                    "branch_probes": sum(
                        item["branch_probes"] for item in gathered_diagnostic_stats
                    ),
                    "eos_branch_candidates_skipped": sum(
                        item["eos_branch_candidates_skipped"]
                        for item in gathered_diagnostic_stats
                    ),
                    "critical_prefix_cache_states": sum(
                        item["critical_prefix_cache_states"]
                        for item in gathered_diagnostic_stats
                    ),
                    "tokens_generated": total_tokens,
                    "rollout_tokens_per_second": total_tokens
                    / max(max_rollout_time, 1e-9),
                    "prompt_sampling_seconds": max(
                        item["sampling_seconds"] for item in gathered_summary
                    ),
                    "rollout_seconds": max_rollout_time,
                    "diagnostic_scoring_seconds": max(
                        item["diagnostic_scoring_seconds"]
                        for item in gathered_diagnostic_stats
                    ),
                    "critical_selection_seconds": max(
                        item["critical_selection_seconds"]
                        for item in gathered_diagnostic_stats
                    ),
                    "branch_probe_seconds": max(
                        item["branch_probe_seconds"]
                        for item in gathered_diagnostic_stats
                    ),
                    "train_forward_backward_seconds": max(
                        item["train_seconds"] for item in gathered_summary
                    ),
                    "optimizer_step_seconds": max(
                        item["optimizer_seconds"] for item in gathered_summary
                    ),
                    "evaluation_seconds": evaluation_seconds,
                    "step_wall_seconds": time.perf_counter() - step_started,
                    "peak_gpu_allocated_gb_max": max(
                        item["peak_allocated_gb"] for item in gathered_summary
                    ),
                    "peak_gpu_reserved_gb_max": max(
                        item["peak_reserved_gb"] for item in gathered_summary
                    ),
                    "peak_gpu_allocated_gb_by_rank": "|".join(
                        f"{item['peak_allocated_gb']:.4f}" for item in gathered_summary
                    ),
                    "peak_gpu_reserved_gb_by_rank": "|".join(
                        f"{item['peak_reserved_gb']:.4f}" for item in gathered_summary
                    ),
                }
                assert self.run_logger is not None
                self.run_logger.log_step(metrics)
                progress.set_postfix(
                    loss=f"{global_loss:.4f}", T=f"{metrics['mean_prompt_T']:.3f}"
                )

        if self.context.is_main:
            assert self.run_logger is not None
            report = save_and_plot_prompt_usage(self.output_dir, self.curriculum)
            generate_all_plots(self.output_dir)
            LOGGER.info("Prompt usage summary: %s", report)
            self.run_logger.close()
        barrier(self.context)
