import copy
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDatasetShard, \
    nested_truncate
import collections
import numpy as np
import torch
from utils import batch_to_device, sample_3d, cal_reward_loss, collate_fn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerState,
)
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import (
    EvalPrediction, speed_metrics, EvalLoopOutput, denumpify_detensorize
)
from transformers.utils import logging

from model import HeroGAN
from training_args_gan import GANTrainingArguments

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

logger = logging.get_logger(__name__)


class GANTrainer(Seq2SeqTrainer):
    def __init__(self, model: Union[HeroGAN, nn.Module] = None, max_seq_length: int = None,
                 warmup_steps: int = None,
                 args: GANTrainingArguments = None,
                 train_samples: Optional = None,
                 eval_samples: Optional = None,
                 candidate_pool: Optional = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None, model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
        super().__init__(model, args, tokenizer=tokenizer, model_init=model_init,
                         compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers)

        self.candidate_embedding = None
        self.warmup_steps = warmup_steps
        self.max_seq_length = max_seq_length
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.candidate_pool = candidate_pool
        self.eval_dataloader = DataLoader(eval_samples, shuffle=False, batch_size=args.per_device_eval_batch_size)
        self.gan_training_state = None

    def get_t5_params(self, forbid):
        params = []
        for n, p in self.model.encoder.named_parameters():
            if forbid not in n:
                params.append((n, p))
        for n, p in self.model.decoder.named_parameters():
            if forbid not in n:
                params.append((n, p))
        for n, p in self.model.lm_head.named_parameters():
            if forbid not in n:
                params.append((n, p))
        params.pop(0)
        return params

    def get_st5_params(self, forbid):
        params = []
        for n, p in self.model.encoder.named_parameters():
            if forbid not in n:
                params.append((n, p))
        for n, p in self.model.project.named_parameters():
            if forbid not in n:
                params.append((n, p))
        for n, p in self.model.norm.named_parameters():
            if forbid not in n:
                params.append((n, p))
        for n, p in self.model.classifier.named_parameters():
            if forbid not in n:
                params.append((n, p))
        return params

    def create_optimizer(self):
        if self.gan_training_state == 'dis':
            param_optimizer = self.get_st5_params('none')

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer_kwargs = {"lr": self.args.st5_pre_lr}

            self.dis_optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        elif self.gan_training_state == 'adv':
            param_optimizer = self.get_t5_params('none')
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_kwargs = {"betas": (self.args.adam_beta1, self.args.adam_beta2),
                                "eps": self.args.adam_epsilon,
                                "lr": self.args.t5_lr}
            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

            param_optimizer = self.get_st5_params('none')
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_kwargs = {"lr": self.args.st5_lr}
            self.dis_optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.gan_training_state == 't5' or self.optimizer is None:
            param_optimizer = self.get_t5_params('none')
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_kwargs = {"betas": (self.args.adam_beta1, self.args.adam_beta2),
                                "eps": self.args.adam_epsilon,
                                "lr": self.args.t5_pre_lr}
            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.gan_training_state == 'gen' or self.gan_training_state == 'adv':
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        else:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.dis_optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def test(self, state='adv'):
        assert state == 'gen' or state == 'adv'
        self.gan_training_state = state
        if state == 'adv':
            self.get_candidate_embedding()
        self.evaluate()

    def train(
            self,
            **kwargs,
    ):

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        args = self.args
        self.is_in_train = True

        # Data loader and number of training steps
        train_samples = self.train_samples

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        num_update_steps_per_epoch = len(train_samples) // args.train_batch_size // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        num_update_steps_per_epoch_dis = len(train_samples) // args.train_batch_size // args.gradient_accumulation_steps
        num_update_steps_per_epoch_dis = max(num_update_steps_per_epoch_dis, 1)

        if args.t5_pre_epochs > 0:
            self.train_t5(train_samples, num_update_steps_per_epoch_dis,
                          total_train_batch_size)

        if args.discriminator_pre_epochs > 0:
            self.pretrain_discriminator(train_samples, num_update_steps_per_epoch_dis,
                                        total_train_batch_size)

        if args.adv_epochs > 0:
            self.adversarial_training(train_samples, num_update_steps_per_epoch,
                                      total_train_batch_size)

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed.\n\n")

    def train_t5(self, train_samples, num_update_steps_per_epoch, total_train_batch_size):
        args = self.args
        self.gan_training_state = 't5'
        self.state = TrainerState()
        gen_mle_max_steps = math.ceil(args.t5_pre_epochs * num_update_steps_per_epoch)
        gen_mle_train_epochs = math.ceil(args.t5_pre_epochs)
        self.create_optimizer_and_scheduler(num_training_steps=gen_mle_max_steps)
        model = self.model

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.per_device_train_batch_size)

        num_examples = (
            self.num_examples(train_dataloader)
        )

        logger.info("***** training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {gen_mle_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {gen_mle_max_steps}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = None
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = gen_mle_max_steps
        self.state.num_train_epochs = gen_mle_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        train_dataloader.collate_fn = self.smart_batching_collate_seq2seq
        tr_loss = torch.tensor(0.0).to(args.device)
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for epoch in range(epochs_trained, gen_mle_train_epochs):

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, None, epoch, None)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, None, epoch, None)

            if self.control.should_training_stop:
                break

    def pretrain_discriminator(self, train_samples, num_update_steps_per_epoch,
                               total_train_batch_size):
        args = self.args
        self.gan_training_state = 'dis'
        self.state = TrainerState()
        dis_pre_max_steps = math.ceil((args.discriminator_pre_epochs) * num_update_steps_per_epoch)
        dis_pre_train_epochs = math.ceil(args.discriminator_pre_epochs)
        self.create_optimizer_and_scheduler(dis_pre_max_steps)
        model = self.model

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.per_device_train_batch_size)

        num_examples = (
            self.num_examples(train_dataloader)
        )

        logger.info("***** Discriminator Pretraining *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {dis_pre_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {dis_pre_max_steps}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.dis_optimizer
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = None
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = dis_pre_max_steps
        self.state.num_train_epochs = dis_pre_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        train_dataloader.collate_fn = self.smart_batching_collate_classify

        tr_loss = torch.tensor(0.0).to(args.device)
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        for epoch in range(epochs_trained, dis_pre_train_epochs):
            epoch_iterator = train_dataloader
            if args.past_index >= 0:
                self._past = None
            steps_in_epoch = (len(epoch_iterator))
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.dis_training_step(model, inputs, is_adv=False)
                tr_loss += tr_loss_step

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, model, None, epoch, None)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            # self._maybe_log_save_evaluate(tr_loss, model, None, epoch, None)
            self.evaluation_step(epoch)

            model.train()

            if self.control.should_training_stop:
                break

    def adversarial_training(self, train_samples, num_update_steps_per_epoch,
                             total_train_batch_size):
        args = self.args
        self.gan_training_state = 'adv'
        self.state = TrainerState()
        adv_max_steps = math.ceil(args.adv_epochs * num_update_steps_per_epoch)
        adv_train_epochs = math.ceil(args.adv_epochs)
        self.create_optimizer_and_scheduler(num_training_steps=adv_max_steps)
        model = self.model

        gen_train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.per_device_train_batch_size)
        train_samples = copy.deepcopy(train_samples)
        dis_train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.per_device_train_batch_size)

        num_examples = (self.num_examples(gen_train_dataloader))

        logger.info("***** Adversarial Training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {adv_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {adv_max_steps}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = gen_train_dataloader
        self.state.trial_name = None
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = adv_max_steps
        self.state.num_train_epochs = adv_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        gen_train_dataloader.collate_fn = self.smart_batching_collate_seq2seq
        dis_train_dataloader.collate_fn = self.smart_batching_collate_classify

        dis_iterator = iter(dis_train_dataloader)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        cnt = 0
        for epoch in range(epochs_trained, adv_train_epochs):
            self.get_candidate_embedding()
            tr_loss = torch.tensor(0.0).to(args.device)
            dis_loss = torch.tensor(0.0).to(args.device)

            epoch_iterator = gen_train_dataloader
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None
            steps_in_epoch = (len(epoch_iterator))
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for gen_step, gen_inputs in enumerate(epoch_iterator):
                cnt += 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                if gen_step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.gen_training_step(model, gen_inputs)
                tr_loss += tr_loss_step

                self.state.global_step += 1
                self.state.epoch = epoch + (gen_step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                if (gen_step + 1) % args.adv_generator_steps == 0:
                    for i in range(args.adv_discriminator_steps):
                        try:
                            dis_inputs = next(dis_iterator)
                        except StopIteration:
                            dis_iterator = iter(dis_train_dataloader)
                            dis_inputs = next(dis_iterator)
                        dis_loss_step = self.dis_training_step(model, dis_inputs, is_adv=True)
                        dis_loss += dis_loss_step

                self._maybe_log_save_evaluate(copy.deepcopy(tr_loss), model, None, epoch, None)
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(copy.deepcopy(tr_loss), model, None, epoch, None)

            model.train()

            logs: Dict[str, float] = {'gen_loss': float(tr_loss), 'dis_loss': float(dis_loss),
                                      'epoch': epoch}
            self.log(logs)

            if self.control.should_training_stop:
                break

    def dis_training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], is_adv: bool):
        features, labels, ground_truth_ids = inputs
        first = features[0]

        feed_in = features.pop()

        if is_adv:
            margin = self.args.margin
            margin_es = self.args.margin_es
            loss_fct = nn.MarginRankingLoss(margin)
            loss_fct_es = nn.MarginRankingLoss(margin_es)
            train_k = self.args.train_k

            second = features[1]
            batch_size, _ = second['input_ids'].shape
            second['input_ids'] = second['input_ids'][:int(batch_size / 2), ]
            second['attention_mask'] = second['attention_mask'][:int(batch_size / 2), ]

            encoder = model.get_encoder()
            first["encoder_outputs"] = encoder(return_dict=True, **first)
            
            tr_loss_step = None

            with torch.no_grad():
                output = model.generate(**feed_in)
                attention_mask = self.model._prepare_attention_mask_for_generation(output,
                                                                                   self.model.config.pad_token_id,
                                                                                   self.model.config.eos_token_id)
                features_fake = {'input_ids': output, 'attention_mask': attention_mask}

            score_real, _ = model.compute_score_matching([first, second])
            score_fake, _ = model.compute_score_matching([first, features_fake])

            y = torch.ones(score_real.shape, device=self.model.device)
            tr_loss_step = loss_fct_es(score_real, score_fake, y)

            candidates, _, _ = self.negative_sampling(second, train_k, ground_truth_ids)

            for can in candidates:
                score_fake, _ = model.compute_score_matching([first, can])
                if tr_loss_step is None:
                    tr_loss_step = loss_fct(score_real, score_fake, y)
                else:
                    tr_loss_step += loss_fct(score_real, score_fake, y)

        else:
            tr_loss_step, _ = model.compute_score_matching(features, labels)

        tr_loss_step.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        self.dis_optimizer.step()
        model.zero_grad()
        return tr_loss_step.detach()

    def gen_training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        pure = inputs.pop('pure')
        batch_to_device(pure, model.device)
        inputs = self._prepare_inputs(inputs)

        inputs_backup = copy.deepcopy(inputs)
        inputs_backup.pop('labels')
        inputs_backup.pop('decoder_input_ids')

        tf_loss_step, outputs = self.compute_loss(model, inputs, return_outputs=True)

        with torch.no_grad():
            encoder = model.get_encoder()
            inputs["encoder_outputs"] = encoder(return_dict=True, **inputs_backup)
        outputs = model(**inputs)
        pg_loss_step = self.policy_gradient(outputs['logits'], inputs, pure, self.model.compute_score_matching)
        tr_loss_step = tf_loss_step + pg_loss_step

        # tr_loss_step = tf_loss_step
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        tr_loss_step.backward()
        self.optimizer.step()
        model.zero_grad()

        return tr_loss_step.detach()

    def policy_gradient(self, out, inputs, pure, cls):
        out = F.softmax(out, dim=-1)
        sample_probs, sample_idx = sample_3d(out)

        with torch.no_grad():
            idx = inputs['decoder_input_ids'].ne(self.tokenizer.pad_token_id).sum(-1)

            tgt = []
            for i, s in zip(idx.cpu(), sample_idx):
                e = torch.arange(len(s))[s.eq(self.tokenizer.eos_token_id)]
                e = e[0] if 0 < len(e) and 4 < e[0] < i else i - 1
                trunc = s[:e].cpu().tolist()
                trunc.append(self.tokenizer.eos_token_id)
                tgt.append(trunc)
            tgt_idx = collate_fn(tgt).to(out.device)
            tgt_mask = self.model._prepare_attention_mask_for_generation(tgt_idx, self.tokenizer.pad_token_id,
                                                                         self.tokenizer.eos_token_id)
            tgt_features = {'input_ids': tgt_idx, 'attention_mask': tgt_mask}
            tgt_reward = cls([pure, tgt_features])[0].detach()

            tgt_reward = 2 * tgt_reward - 1

        loss_sc = cal_reward_loss(sample_probs, tgt_reward, idx)

        return loss_sc

    @torch.no_grad()
    def get_candidate_embedding(self):
        candidate_pool = self.candidate_pool
        batches = []
        candidate_embedding = []
        for i in range(len(candidate_pool)):
            if i % self.args.per_device_train_batch_size == 0:
                if i > 0:
                    batches.append(batch)
                batch = []
            batch.append(candidate_pool[i])
        batches.append(batch)
        for batch in batches:
            batch = self.tokenize(batch)
            batch_to_device(batch, self.model.device)
            candidate_embedding.append(self.model.get_sentence_embedding(batch)['sentence_embedding'])
        candidate_embedding = torch.cat(candidate_embedding, dim=0)
        self.candidate_embedding = candidate_embedding

    @torch.no_grad()
    def negative_sampling(self, response, k, ground_truth_ids):
        candidate_embedding = self.candidate_embedding
        candidate_pool = self.candidate_pool
        target_embedding = self.model.get_sentence_embedding(response)['sentence_embedding']
        neg_embedding = []
        indexes = []
        i = 0
        for embedding in target_embedding:
            if self.args.use_euclidean_distance:
                distance = F.pairwise_distance(embedding, candidate_embedding)
                index = torch.topk(distance, k + 1, largest=False, sorted=True)[1]
            else:
                sim = torch.cosine_similarity(embedding, candidate_embedding, dim=-1)
                index = torch.topk(sim, k + 1, largest=True, sorted=True)[1]
            if ground_truth_ids is not None:
                where = torch.nonzero(index != ground_truth_ids[i], as_tuple=False)
                index = torch.index_select(index, dim=0, index=where.squeeze())
            index = index[:k]
            indexes.append(index)
            neg_embedding.append(candidate_embedding[index])
            i += 1
        neg_embedding = torch.stack(neg_embedding, 0)
        indexes = torch.stack(indexes, 0).permute(1, 0)
        neg_sentences = []
        for index in indexes:
            batch = []
            for idx in index:
                batch.append(candidate_pool[idx])
            tmp = self.tokenize(batch)
            batch_to_device(tmp, self.model.device)
            neg_sentences.append(tmp)

        return neg_sentences, neg_embedding.permute(1, 0, 2), indexes.permute(1, 0)

    @torch.no_grad()
    def sentence_t5_ranking(self, context, sentences, indexes):
        def take_first(e):
            return e[0]

        indexes = indexes.cpu()

        scores = []
        batch_size, _ = context['input_ids'].shape
        for sentence in sentences:
            labels = torch.ones(batch_size, device=self.model.device).long()
            _, score = self.model.compute_score_matching([context, sentence], labels)
            # score = score[:, 1]
            scores.append(score)
        scores = torch.stack(scores).squeeze(-1).cpu()
        top = torch.argmax(scores, 0)
        result = []
        i = 0
        for idx in top:
            result.append(
                self._pad_tensors_to_max_len(sentences[idx]['input_ids'][i].unsqueeze(0), self.max_seq_length).squeeze(
                    0))
            i += 1
        result = torch.stack(result)

        seq2seq = 0

        for i in range(batch_size):
            if int(indexes[i][top[i]]) == -1:
                seq2seq += 1

        scores = scores.permute(1, 0)
        ordered = []
        for i in range(scores.shape[0]):
            tmp = set()
            for j in range(scores.shape[1]):
                if int(indexes[i][j]) != -1:
                    tmp.add((float(scores[i][j]), int(indexes[i][j])))
            tmp = list(tmp)
            tmp.sort(key=take_first, reverse=True)
            for j in range(len(tmp)):
                tmp[j] = tmp[j][1]
            ordered.append(tmp)
        return result, ordered, seq2seq

    def smart_batching_collate_seq2seq(self, batch):
        features = []
        ground_truth_ids = []

        eval_k_es = self.args.eval_k_es
        train_k_es = self.args.train_k_es
        prob = self.args.es_truth_ratio

        es_samples = []
        es_ids = []

        features_pure = []

        for example in batch:
            feed_in = example.texts[0]
            ess = example.texts[3][:train_k_es]
            if self.model.training and random.random() < prob:
                found = False
                for item in ess:
                    if example.texts[1]['id'] == item['id']:
                        found = True
                if not found:
                    ess[random.randint(0, train_k_es-1)] = example.texts[1]
            for i in range(train_k_es):
                feed_in += (' candidate: ' + ess[i]['utterance'])
            feature = self.tokenizer(feed_in, max_length=self.max_seq_length, truncation=True)
            feature['labels'] = \
                self.tokenizer(example.texts[1]['utterance'], max_length=self.max_seq_length, truncation=True)[
                    "input_ids"]
            ground_truth_ids.append(example.texts[1]['id'])
            feature_pure = self.tokenizer(example.texts[0], max_length=self.max_seq_length, truncation=True)
            features_pure.append(feature_pure)
            features.append(feature)

        if self.gan_training_state == 'adv_eval':
            for i in range(eval_k_es):
                samples = []
                ids = []
                for example in batch:
                    samples.append(example.texts[3][i]['utterance'])
                    ids.append(int(example.texts[3][i]['id']))
                sample_features = self.tokenize(samples)
                batch_to_device(sample_features, self.model.device)
                es_samples.append(sample_features)
                es_ids.append(ids)
            es_ids = torch.tensor(es_ids, dtype=torch.long, device=self.model.device)

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            i = 0
            for feature in features:
                remainder = [-100] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                features_pure[i]["labels"] = feature["labels"]
                i += 1

        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        ).data

        features_pure = self.tokenizer.pad(
            features_pure,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        ).data

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        if self.gan_training_state == 'adv_eval':
            features["ground_truth_ids"] = ground_truth_ids
            features["es_samples"] = es_samples
            features["es_ids"] = es_ids.permute(1, 0)

        if self.gan_training_state == 'adv_eval' or self.gan_training_state == 'adv':
            features['pure'] = features_pure

        return features

    def smart_batching_collate_classify(self, batch):
        train_k_es = self.args.train_k_es
        prob = self.args.es_truth_ratio
        num_texts = 3
        texts = [[] for _ in range(num_texts)]
        labels = []

        ground_truth_ids = []

        for example in batch:
            feed_in = example.texts[0]
            ess = example.texts[3][:train_k_es]
            if self.model.training and random.random() < prob:
                found = False
                for item in ess:
                    if example.texts[1]['id'] == item['id']:
                        found = True
                if not found:
                    ess[random.randint(0, train_k_es-1)] = example.texts[1]
            for i in range(train_k_es):
                feed_in += (' candidate: ' + ess[i]['utterance'])
            texts[0].append(example.texts[0])
            # texts[1].append(example.texts[1])
            texts[1].append(example.texts[1]['utterance'])
            texts[2].append(feed_in)
            ground_truth_ids.append(example.texts[1]['id'])
            labels.append(1)

        for example in batch:
            texts[1].append(example.texts[2]['utterance'])
            labels.append(0)

        labels = torch.tensor(labels).to(self.model.device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            batch_to_device(tokenized, self.model.device)
            sentence_features.append(tokenized)

        return sentence_features, labels, ground_truth_ids

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt",
                                     max_length=self.max_seq_length))
        return output

    @torch.no_grad()
    def evaluation_step(self, epoch: int = -1, steps: int = -1) -> float:
        self.model.eval()
        total = 0
        correct = 0

        true_pos = 0
        false_pos = 0
        false_neg = 0

        eval_dataloader = self.eval_dataloader

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the dataset" + out_txt)
        eval_dataloader.collate_fn = self.smart_batching_collate_classify
        # mismatch = []
        for step, batch in enumerate(eval_dataloader):
            features, label_ids, _ = batch
            features = [features[0], features[1]]
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], self.model.device)
            label_ids = label_ids.to(self.model.device)
            _, prediction = self.model.compute_score_matching(features, labels=None)

            total += prediction.size(0)

            prediction = prediction.squeeze(1)
            # ne = (prediction > 0).long().ne(label_ids)
            true_pos += (prediction[:prediction.size(0) // 2] > 0).long().eq(
                label_ids[:prediction.size(0) // 2]).sum().item()
            false_neg += (prediction[:prediction.size(0) // 2] > 0).long().eq(
                label_ids[prediction.size(0) // 2:]).sum().item()
            false_pos += (prediction[prediction.size(0) // 2:] > 0).long().eq(
                label_ids[:prediction.size(0) // 2]).sum().item()
            # ne = torch.argmax(prediction, dim=1).ne(label_ids)
            # true_pos += torch.argmax(prediction[:prediction.size(0) // 2], dim=1).eq(
            #     label_ids[:prediction.size(0) // 2]).sum().item()
            # false_neg += torch.argmax(prediction[:prediction.size(0) // 2], dim=1).eq(
            #     label_ids[prediction.size(0) // 2:]).sum().item()
            # false_pos += torch.argmax(prediction[prediction.size(0) // 2:], dim=1).eq(
            #     label_ids[:prediction.size(0) // 2]).sum().item()

            # for i in range(ne.shape[0]):
            #     if ne[i] == 1:
            #         mismatch.append({'x': texts[0][i], 'y': texts[1][i], 'label': int(label_ids[i])})
            correct += (prediction > 0).long().eq(label_ids).sum().item()
            # correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        accuracy = correct / total

        # with open('mismatch.json', 'w') as f:
        #     json.dump(mismatch, f, indent='')

        precision = true_pos / (true_pos + false_neg)
        recall = true_pos / (true_pos + false_pos)
        f1 = 2 * (precision * recall) / (precision + recall)

        logs: Dict[str, float] = {'acc': float(accuracy), 'f1': float(f1), 'epoch': epoch, 'correct': float(correct),
                                  'total': float(total)}
        self.log(logs)

        return accuracy

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if self.gan_training_state != 'adv_eval':
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        batch_size = inputs['input_ids'].shape[0]

        ground_truth_ids = inputs.pop('ground_truth_ids')
        es_samples = inputs.pop('es_samples')
        es_ids = inputs.pop('es_ids')

        pure = inputs.pop('pure')
        batch_to_device(pure, self.model.device)

        pad_token_id = self.model.config.pad_token_id
        eos_token_id = self.model.config.eos_token_id

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": False,
        }

        eval_sample_beams = self.args.eval_sample_beams

        candidates = []

        indexes = []

        for i in range(eval_sample_beams):
            if i == 0:
                generated_tokens = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
            else:
                generated_tokens = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                    do_sample=True
                )

            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

            attention_mask = self.model._prepare_attention_mask_for_generation(generated_tokens, pad_token_id,
                                                                               eos_token_id)

            generated_tokens = {"input_ids": generated_tokens, 'attention_mask': attention_mask}

            fake_index = -torch.ones([batch_size, 1], dtype=torch.long, device=self.model.device)

            with torch.no_grad():
                candidates_tmp, _, indexes_tmp = self.negative_sampling(generated_tokens, self.args.eval_k, None)
                indexes.append(indexes_tmp)
                indexes.append(fake_index)
                candidates.extend(candidates_tmp)
                candidates.append(generated_tokens)

        indexes.append(es_ids)
        candidates.extend(es_samples)

        indexes = torch.cat(indexes, 1)
        with torch.no_grad():
            result, rank, seq2seq = self.sentence_t5_ranking(pure, candidates, indexes)

        hit_1 = 0
        hit_5 = 0
        hit_10 = 0
        hit_25 = 0
        hit_50 = 0

        for i in range(len(ground_truth_ids)):
            if ground_truth_ids[i] in rank[i][:50]:
                hit_50 += 1
                if ground_truth_ids[i] in rank[i][:25]:
                    hit_25 += 1
                    if ground_truth_ids[i] in rank[i][:10]:
                        hit_10 += 1
                        if ground_truth_ids[i] in rank[i][:5]:
                            hit_5 += 1
                            if ground_truth_ids[i] == rank[i][0]:
                                hit_1 += 1

        hit = {'hit_1': hit_1, 'hit_5': hit_5, 'hit_10': hit_10, 'hit_25': hit_25, 'hit_50': hit_50, 'seq2seq': seq2seq}

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (hit, result, labels)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:

        state = self.gan_training_state
        if state == 'adv':
            self.gan_training_state = "adv_eval"

        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        self._memory_tracker.start()

        eval_dataloader = self.eval_dataloader
        eval_dataloader.collate_fn = self.smart_batching_collate_seq2seq
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        self.gan_training_state = state

        return output.metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop

        total = 0
        hit_1 = 0
        hit_5 = 0
        hit_10 = 0
        hit_25 = 0
        hit_50 = 0
        seq2seq = 0

        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            total += batch_size

            # Prediction step
            if self.gan_training_state == 'adv_eval':
                hit, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                           ignore_keys=ignore_keys)
                hit_1 += hit['hit_1']
                hit_5 += hit['hit_5']
                hit_10 += hit['hit_10']
                hit_25 += hit['hit_25']
                hit_50 += hit['hit_50']
                seq2seq += hit['seq2seq']

                loss = None

            else:
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.gan_training_state == 'adv_eval':
            self.log(
                {'hit@1': hit_1 / total, 'hit@5': hit_5 / total, 'hit@10': hit_10 / total, 'hit@25': hit_25 / total,
                 'hit@50': hit_50 / total, 'seq2seq': seq2seq / total})

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
