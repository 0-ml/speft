import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import accelerate
import datasets as hf_datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.sparse_peft import sparsify_model, update_linear_sparse
from model.utils import _load_cached_dataset
from sparse_args import SparseArguments


# GLUE tasks:
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"), 
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use the slow or fast tokenizer"},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore mismatched sizes between the model and the tokenizer"
        },
    )

@dataclass
class DataArguments:
    task_name: str = field(
        metadata={
            "help": "The name of the task to train or evaluate. One of 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'"
        },
    )
    max_length: int = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # This class is created to override the default values of the TrainingArguments
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."},
    )
    num_train_epochs: float = field(
        default=30,
        metadata={"help": "Total number of training epochs to perform."},
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    gradient_checkpointing: bool = field(
        default=False
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Adam."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "A seed for reproducible training."},
    )
    output_dir: str = field(
        default="results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to adopt."},
    )
    save_total_limit: int = field(
        default=1,
        metadata={
            "help": "Total number of times the model was saved."},
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    run_name: str = field(default="none")
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio * total_steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply."},
    )
    task_type: str = field(
        default="none",
    )


def create_dataset(
    args,
    logger,
    model,
    tokenizer,
    raw_datasets,
    accelerator,
    is_regression,
    label_list,
    num_labels,
    config,
):
    # create subset of glue dataset
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id
        != transformers.PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    return processed_datasets

def trainable_parameters(model, decay):
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return params

def main():
    """
    CLI

    Example: full fine-tune bert-base-uncased on sst2

    ```bash
    accelerate launch --multi_gpu cls_train.py --model_name_or_path bert-base-uncased --task_name sst2
    ```
    """
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SparseArguments)
    )
    args = parser.parse_args_into_dataclasses()
    args_dict = {}
    for a in args:
        args_dict.update(vars(a))
    args = argparse.Namespace(**args_dict)

    if args.output_dir and args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=args.output_dir
    )

    if accelerator.is_local_main_process:
        hf_datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        hf_datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # raw datasets
    raw_datasets = _load_cached_dataset("glue", args.task_name)

    # model and tokenizer
    is_regression = args.task_name == "stsb"
    label_list = []

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
    )
    # model for full fine-tuning is created here
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    
    # preprocess datasets
    processed_datasets = create_dataset(
        args=args,
        logger=logger,
        model=model,
        tokenizer=tokenizer,
        raw_datasets=raw_datasets,
        accelerator=accelerator,
        is_regression=is_regression,
        label_list=label_list,
        num_labels=num_labels,
        config=config,
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched" if args.task_name == "mnli" else "validation"
    ]

    # data collator
    if args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    else:
        data_collator = transformers.DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # initialize wandb
    if args.report_to:
        entity = os.environ.get("WANDB_ENTITY", None)
        # if entity is None:
        #     raise ValueError("Please set WANDB_ENTITY environment variable.")
        model_name = args.model_name_or_path.split("/")[-1]
        project_name = f"glue-{model_name}"
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers(
            project_name, experiment_config,
            init_kwargs={
                "wandb": {
                    "entity": entity,
                    "name": args.run_name,
                },
            }
        )

    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj", # opt llama
        "query", "value", # bert roberta
    ]
    
    model = sparsify_model(accelerator, model, train_dataloader, target_modules, args)
    
    if not args.lora_enable and not args.sparse_enable:
        for p in model.parameters():
            p.requires_grad = True
                
    # print number of trainable parameters
    num_trainable = num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_trainable += p.nelement()
        num_params += p.nelement()
    trainable_perc = num_trainable / num_params
    print(f'{num_trainable=}, {num_params=}, {trainable_perc=:.2%}')
    set_seed(args.seed)

    # optimizer
    train_params = trainable_parameters(model, args.weight_decay)
    optimizer = torch.optim.AdamW(train_params, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_steps == -1:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr scheduler
    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    # metric object
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    progress_bar = tqdm(
        range(int(args.max_steps)), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # resume from checkpoint
    if args.resume_from_checkpoint:
        ckpt_path = Path(args.resume_from_checkpoint)
        dirs = [f for f in ckpt_path.iterdir() if f.is_dir()]
        # sort dirs by creation time
        dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        ckpt = dirs[0]
        accelerator.print(f"Loading checkpoint from {ckpt}")
        accelerator.load_state(ckpt)

        training_difference = ckpt.name
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.removeprefix("epoch_")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = (
                int(training_difference.removeprefix("step_"))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step = resume_step % len(train_dataloader)

        progress_bar.update(completed_steps)

    # sequence classification training loop
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.report_to:
            total_loss = 0

        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            active_dataloader = accelerate.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            ouptuts = model(**batch)
            loss = ouptuts.loss

            if args.report_to:
                total_loss += loss.detach().float()
                # accelerator.log({"train_loss": loss.detach().float()})

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(active_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if (
                args.sparse_mask_update_steps > 0
                and (step+1) % args.sparse_mask_update_steps == 0
                and args.sparse_enable
            ):
                update_linear_sparse(accelerator, model, train_dataloader, args)
                # reset state as we use a new mask
                optimizer.optimizer.param_groups.clear()
                optimizer.optimizer.state.clear()
                for g in trainable_parameters(model, args.weight_decay):
                    optimizer.optimizer.add_param_group(g)

            # save accelerator state
            if (
                args.output_dir is not None
                and args.save_strategy == "steps"
                and completed_steps % args.save_steps == 0
            ):
                output_dir = Path(args.output_dir) / f"step_{completed_steps}"
                accelerator.save_state(output_dir)

            if completed_steps >= args.max_steps:
                break

        # evaluate on validation set
        model.eval()
        samples_seen = 0
        active_dataloader = eval_dataloader
        for step, batch in enumerate(active_dataloader):
            with torch.no_grad():
                ouptuts = model(**batch)
            predictions = (
                ouptuts.logits.argmax(dim=-1)
                if not is_regression
                else ouptuts.logits.squeeze()
            )
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    references = references[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += references.shape[0]

            metric.add_batch(predictions=predictions, references=references)
        eval_metric = metric.compute()
        logger.info(f"epoch {epoch} eval_metric: {eval_metric}")

        if args.report_to:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "total_train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.save_strategy == "epoch":
            output_dir = Path(args.output_dir) / f"epoch_{epoch}"
            accelerator.save_state(output_dir)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        eval_metric = metric.compute()
        accelerator.log({"mnli-mm": eval_metric})
        logger.info(f"mnli-mm: {eval_metric}")
    
    if args.report_to:
        accelerator.end_training()
        
    # save the last checkpoint
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        save_path = Path(args.output_dir) / "all_results.json"
        with open(save_path, "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    """
    transformers.TrainingArguments has hyperparameters related to training,
    override the default values by passing them as command line arguments,
    or modify the default values in the TrainingArguments class in this file.

    Example usage:

    Full fine-tune bert-base-uncased on sst2, DDP, report to wandb, save to ./results

    ```bash
    accelerate launch --multi_gpu cls_train.py --model_name_or_path bert-base-uncased --task_name sst2
    ```

    """
    main()