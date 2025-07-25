import contextlib
import functools
import math
import os
import tempfile

import peft
import torch
import tqdm
from accelerate import Accelerator
from torch.func import functional_call, vmap
from torch.func import grad as grad_fn
from torch.nn import functional as F
# from transformers.models.gemma2.modeling_gemma2 import Gemma2SdpaAttention
# from transformers.models.llama.modeling_llama import LlamaSdpaAttention
# from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention

from layers import LinearSparse


_random_init_funcs = {}


def disable_random_init() -> None:
    do_nothing = lambda *args, **kwargs: None
    for name in dir(torch.nn.init):
        if name.startswith('_'):
            continue
        func = getattr(torch.nn.init, name)
        if not callable(func):
            continue
        _random_init_funcs[name] = func
        setattr(torch.nn.init, name, do_nothing)

def enable_random_init() -> None:
    for name, func in _random_init_funcs.items():
        setattr(torch.nn.init, name, func)

@contextlib.contextmanager
def disable_random_init_context():
    disable_random_init()
    yield
    enable_random_init()

def train_step(model, batch, temperature=1.0):
    def forward(self, input, target):
        return F.cross_entropy(
            input / temperature, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction=self.reduction,
            label_smoothing=self.label_smoothing)
    old_forward = torch.nn.CrossEntropyLoss.forward
    torch.nn.CrossEntropyLoss.forward = forward
    outputs = model(**batch)
    loss = outputs["loss"]
    torch.nn.CrossEntropyLoss.forward = old_forward
    return loss

def compute_random_scores(
    accelerator: Accelerator,
    model: torch.nn.Module,
    linear_sparse_layers,
    train_dataloader,
    steps,
):
    scores = []
    for l in linear_sparse_layers:
        scores.append(torch.rand_like(l.weight.flatten()))
    return scores

def compute_magnitude_scores(
    accelerator: Accelerator,
    model: torch.nn.Module,
    linear_sparse_layers,
    train_dataloader,
    steps,
):
    scores = []
    for l in linear_sparse_layers:
        scores.append(l.weight.detach().flatten().abs())
    return scores

def compute_gradient_based_scores(
    func,
    accelerator: Accelerator,
    model: torch.nn.Module,
    linear_sparse_layers,
    train_dataloader,
    steps,
):
    train_dataloader = accelerator.prepare(train_dataloader)
    train_iterator = iter(train_dataloader)

    grads = [torch.zeros_like(layer.weight) for layer in linear_sparse_layers]
    for step in tqdm.trange(steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        model.zero_grad()
        loss = train_step(model, batch)
        loss = loss / steps
        accelerator.backward(loss)

        for layer_idx, layer in enumerate(linear_sparse_layers):
            grads[layer_idx] += layer.weight.grad

    scores = [
        func(
            layer.weight.detach().flatten(),
            grad.detach().flatten())
        for layer, grad in zip(linear_sparse_layers, grads)]
    # norm_factor = sum(x.sum() for x in scores)
    # return [s / norm_factor for s in scores]
    return scores

# Hack LlamaSdpaAttention.forward to use super().forward instead
# def hack_forward(model_name):
#     if "Qwen" in model_name:
#         def hacked_forward(self, *args, **kwargs):
#             return super(Qwen2SdpaAttention, self).forward(*args, **kwargs)
#         sdpa_forward = Qwen2SdpaAttention.forward
#         Qwen2SdpaAttention.forward = hacked_forward
#     elif "gemma" in model_name:
#         def hacked_forward(self, *args, **kwargs):
#             return super(Gemma2SdpaAttention, self).forward(*args, **kwargs)
#         sdpa_forward = Gemma2SdpaAttention.forward
#         Gemma2SdpaAttention.forward = hacked_forward
#     else:
#         def hacked_forward(self, *args, **kwargs):
#             return super(LlamaSdpaAttention, self).forward(*args, **kwargs)
#         sdpa_forward = LlamaSdpaAttention.forward
#         LlamaSdpaAttention.forward = hacked_forward
#     return sdpa_forward

# # Revert hacked forward
# def Revert_hacked_forward(model_name, sdpa_forward):
#     if "Qwen" in model_name:
#         Qwen2SdpaAttention.forward = sdpa_forward
#     elif "gemma" in model_name:
#         Gemma2SdpaAttention.forward = sdpa_forward
#     else:
#         LlamaSdpaAttention.forward = sdpa_forward

def compute_loss(params, sample, attention_mask, target, model, temperature=1.0):
    def forward(self, input, target):
        return F.cross_entropy(
            input / temperature, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction=self.reduction,
            label_smoothing=self.label_smoothing)
    old_forward = torch.nn.CrossEntropyLoss.forward
    torch.nn.CrossEntropyLoss.forward = forward
    batch = sample.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    labels = target.unsqueeze(0)
    predictions = functional_call(model, params, (batch, attention_mask, labels))

    if 'Qwen' in model.name_or_path:
        return predictions.loss
    elif 'gemma' in model.name_or_path:
        loss_fn = torch.nn.CrossEntropyLoss()
        shift_logits = predictions.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_fn(shift_logits, shift_labels)
        torch.nn.CrossEntropyLoss.forward = old_forward
        return loss
    else:
        if model.config.problem_type is None:
            if model.num_labels == 1:
                model.config.problem_type = "regression"
            elif model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                model.config.problem_type = "single_label_classification"
            else:
                model.config.problem_type = "multi_label_classification"
        if model.config.problem_type == "regression":
            loss_fn = torch.nn.MSELoss()
        elif model.config.problem_type == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
        elif model.config.problem_type == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()

    loss = loss_fn(predictions.logits, labels)
    torch.nn.CrossEntropyLoss.forward = old_forward
    return loss

def compute_sample_grads(model, params, sample, mask, targets):
    """ manually process each sample with per sample gradient """
    ft_compute_loss = functools.partial(compute_loss, model=model)
    ft_compute_grad = grad_fn(ft_compute_loss)
    sample_grads = [ft_compute_grad(params, sample[i], mask[i], targets[i]) for i in range(targets.size(0))]
    ft_per_sample_grads = {}
    for sample_grad in sample_grads:
        for (name, grad) in sample_grad.items():
            if name not in ft_per_sample_grads:
                ft_per_sample_grads[name] = []
            ft_per_sample_grads[name].append(grad)
    for name in ft_per_sample_grads:
        ft_per_sample_grads[name] = torch.stack(ft_per_sample_grads[name])
    return ft_per_sample_grads

def compute_fisher_info_scores(
    accelerator: Accelerator,
    model: torch.nn.Module,
    linear_sparse_layers,
    train_dataloader,
    steps,
):
    train_dataloader = accelerator.prepare(train_dataloader)
    train_iterator = iter(train_dataloader)

    scores = [torch.zeros_like(layer.weight) for layer in linear_sparse_layers]
    # params = [layer.weight for layer in linear_sparse_layers]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    num_examples = 0

    # Hack LlamaSdpaAttention.forward to use super().forward instead
    # sdpa_forward = hack_forward(model.name_or_path)
    # def hack_forward(self, *args, **kwargs):
    #     return super(Gemma2SdpaAttention, self).forward(*args, **kwargs)
    # sdpa_forward, Gemma2SdpaAttention.forward = \
    #     Gemma2SdpaAttention.forward, hack_forward

    for step in tqdm.trange(steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        model.zero_grad()
        # params = {k: v.detach() for k, v in model.named_parameters()}
        # buffers = {k: v.detach() for k, v in model.named_buffers()}
        if 'bert' in model.name_or_path or 'Qwen' in model.name_or_path or 'gemma' in model.name_or_path:
            ft_per_sample_grads = compute_sample_grads(model, params, batch['input_ids'], batch['attention_mask'], batch['labels'])
        else:
            ft_compute_loss = functools.partial(compute_loss, model=model)
            ft_compute_grad = grad_fn(ft_compute_loss)
            ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
            ft_per_sample_grads = ft_compute_sample_grad(params, batch['input_ids'], batch['labels'])
        #  fisher_info = 0
        #  for grad in ft_per_sample_grads:
            #  fisher_info += grad ** 2
        #  fisher_info /= len(ft_per_sample_grads)
        for score, grad in zip(scores, ft_per_sample_grads):
            score += (ft_per_sample_grads[grad] ** 2).sum(0)
        num_examples += batch["input_ids"].shape[0]

        # for layer_idx, layer in enumerate(linear_sparse_layers):
        #     grads[layer_idx] += layer.weight.grad

    scores = [s.detach().flatten().cpu() / num_examples for s in scores]

    # scores = [
    #     func(
    #         layer.weight.detach().flatten().cpu(),
    #         grad.detach().flatten().cpu())
    #     for layer, grad in zip(linear_sparse_layers, grads)]
    # norm_factor = sum(x.sum() for x in scores)
    # return [s / norm_factor for s in scores]

    # Revert hacked forward
    # Revert_hacked_forward(model.name_or_path, sdpa_forward)
    # Gemma2SdpaAttention.forward = sdpa_forward

    return scores

def compute_synflow_scores(
    accelerator: Accelerator,
    model: torch.nn.Module,
    linear_sparse_layers,
    train_dataloader,
    steps,
):
    train_dataloader = accelerator.prepare(train_dataloader)
    train_iterator = iter(train_dataloader)

    old_activations = {}
    activation_names = [
        'threshold', 'relu', 'rrelu', 'hardtanh', 'relu6', 'sigmoid', 'hardsigmoid', 'tanh',
        'silu', 'mish', 'hardswish', 'elu', 'celu', 'selu', 'glu', 'gelu', 'hardshrink', 'leaky_relu',
        'logsigmoid', 'softplus', 'softshrink', 'prelu', 'softsign', 'tanhshrink',
        'softmin', 'softmax', 'log_softmax'
    ]

    def identity(x, *args, **kwargs):
        return x

    @torch.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        # disable nonlinearity
        for a in activation_names:
            old_activations[a] = getattr(torch.nn.functional, a)
            setattr(torch.nn.functional, a, identity)
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])
        # reenable all nonlinearity
        for a in activation_names:
            setattr(torch.nn.functional, a, old_activations.pop(a))

    signs = linearize(model)

    batch = next(train_iterator)
    input_ids = batch["input_ids"]
    inputs_embeds = model.get_input_embeddings()(input_ids)
    ones = torch.ones_like(inputs_embeds[:1, :1])
    outputs = model(inputs_embeds=ones)
    outputs['logits'].sum().backward()

    scores = []
    for layer in linear_sparse_layers:
        p = layer.weight
        scores.append(torch.clone(p.grad * p).detach().abs_())
    nonlinearize(model, signs)
    model.zero_grad()
    return scores

def compute_grasp_scores(
    accelerator: Accelerator,
    model: torch.nn.Module,
    linear_sparse_layers,
    train_dataloader,
    steps,
):
    train_dataloader = accelerator.prepare(train_dataloader)
    train_iterator = iter(train_dataloader)
    # first gradient vector without computational graph
    stopped_grads = 0
    temperature = 200 # ?

    # Hack LlamaSdpaAttention.forward to use super().forward instead
    # sdpa_forward = hack_forward(model.name_or_path)
    # def hack_forward(self, *args, **kwargs):
    #     return super(Gemma2SdpaAttention, self).forward(*args, **kwargs)
    # sdpa_forward, Gemma2SdpaAttention.forward = \
    #     Gemma2SdpaAttention.forward, hack_forward

    model.zero_grad()
    for b in tqdm.trange(steps):
        batch = next(train_iterator)
        loss = train_step(model, batch, temperature)
        loss = loss / steps
        params = [l.weight for l in linear_sparse_layers]
        grads = torch.autograd.grad(loss, params, create_graph=False)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        stopped_grads += flatten_grads
    # move weights to cpu to save gpu memory
    stopped_grads = stopped_grads.cpu()
    # torch.cuda.empty_cache()

    model.zero_grad()
    for b in tqdm.trange(steps):
        batch = next(train_iterator)
        loss = train_step(model, batch, temperature)
        loss = loss / steps
        params = [l.weight for l in linear_sparse_layers]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        # move weights to cpu to save gpu memory
        flatten_grads = torch.cat(
            [g.cpu().reshape(-1) for g in grads if g is not None])
        gnorm = (stopped_grads * flatten_grads).sum()
        gnorm.backward()

    scores = [torch.zeros_like(layer.weight) for layer in linear_sparse_layers]
    for layer_idx, layer in enumerate(linear_sparse_layers):
        score = layer.weight.grad * layer.weight.data
        scores[layer_idx] = torch.clone(score).detach()

    # all_scores = torch.cat([torch.flatten(v) for v in scores])
    # norm = torch.abs(torch.sum(all_scores)) + 1e-10
    # for layer_idx, layer in enumerate(linear_sparse_layers):
    #     scores[layer_idx].div_(norm)

    # Revert hacked forward
    # Revert_hacked_forward(model.name_or_path, sdpa_forward)
    # Gemma2SdpaAttention.forward = sdpa_forward

    return scores

def update_linear_sparse(accelerator, model, trainloader, args):
    layers = [layer for layer in model.modules() if isinstance(layer, LinearSparse)]
    if not any(layers):
        return
    method = layers[0].sparse_method
    score_funcs = {
        'random': compute_random_scores,
        'magnitude': compute_magnitude_scores,
        'gradient': functools.partial(
            compute_gradient_based_scores, lambda w, s: torch.abs(s)),
        'snip': functools.partial(
            compute_gradient_based_scores, lambda w, s: torch.abs(w * s)),
        'force': functools.partial(
            compute_gradient_based_scores, lambda w, s: -w * s),
        'Taylor-FO': functools.partial(
            compute_gradient_based_scores, lambda w, s: (w * s) ** 2),
        'synflow': compute_synflow_scores,
        'grasp': compute_grasp_scores,
        'fisher_info': compute_fisher_info_scores,
    }
    linear_sparse_layers = [
        layer for layer in model.modules() if isinstance(layer, LinearSparse)
    ]
    for layer in linear_sparse_layers:
        layer.apply_adapter_weight()
        layer.use_original_linear(True)
    steps = int(math.ceil(
        args.sparse_score_accumulation_examples /
        args.per_device_train_batch_size))
    scores_dir = os.path.join('tmp', 'speft', args.run_name)
    os.makedirs(scores_dir, exist_ok=True)
    scores_path = os.path.join(
        scores_dir, f'scores-{accelerator.process_index}.pt')
    if not os.path.exists(scores_path):
        scores = score_funcs[method](
            accelerator, model, linear_sparse_layers, trainloader, steps)
        torch.save(scores, scores_path)
    accelerator.wait_for_everyone()
    rank_scores = [
        torch.load(os.path.join(scores_dir, f'scores-{r}.pt'), map_location='cpu')
        for r in range(accelerator.num_processes)]
    scores = [
        sum([s[l] for s in rank_scores]) for l in range(len(rank_scores[0]))]
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     for r in range(accelerator.num_processes):
    #         path = os.path.join(scores_dir, f'scores-{r}.pt')
    #         if os.path.exists(path):
    #             os.remove(path)
    print('Scores done')
    if args.sparse_global_topk:
        all_scores = torch.cat([torch.flatten(v) for v in scores])
        if 0 < args.sparse_k <= 1:
            k = int(args.sparse_k * all_scores.numel())
        else:
            k = int(args.sparse_k)
        _, all_index = torch.topk(all_scores, k, sorted=True)
        pos = 0
        num_selected = num_params = 0
        for i, layer in enumerate(linear_sparse_layers):
            layer.use_original_linear(False)
            size = layer.weight.nelement()
            mask = torch.greater_equal(all_index, pos)
            mask &= torch.less(all_index, pos + size)
            layer_mask = all_index[mask] - pos
            layer.update_weight_selection_by_mask(
                layer.active_adapter, layer_mask)
            layer_num_selected = layer_mask.nelement()
            accelerator.log(
                {
                    f'mask/sparse_{i}/percent': layer_num_selected / size,
                    f'mask/sparse_{i}/num': layer_num_selected
                },
                log_kwargs={
                    "wandb": {
                        "commit": False,
                    },
                }
            )
            num_selected += layer_num_selected
            num_params += size
            pos += size
    else:
        for layer, score in zip(linear_sparse_layers, scores):
            layer.use_original_linear(False)
            if 0 < args.sparse_k <= 1:
                layer_k = int(args.sparse_k * score.numel())
            else:
                layer_k = int(args.sparse_k) / len(linear_sparse_layers)
            layer.update_weight_selection_by_topk_score(
                layer.active_adapter, score, layer_k)
    return linear_sparse_layers

def replace_layers(parent, replace_func, parent_name=''):
    for name, module in parent.named_children():
        module_name = parent_name + ('.' if parent_name else '') + name
        replace_layers(module, replace_func, module_name)
        replace_module = replace_func(module, module_name)
        if replace_module is not None:
            setattr(parent, name, replace_module)

def replace_linear(module, name, sparse_keys, args):
    if not isinstance(module, torch.nn.Linear):
        return None
    if not any(name.endswith(f'.{k}') for k in sparse_keys):
        return None
    sparse_module = LinearSparse(
        module.in_features, module.out_features,
        sparse_method=args.sparse_method,
        sparse_alpha=args.sparse_alpha,
        sparse_dropout=args.sparse_dropout,
        bias=module.bias is not None)
    sparse_module.to(module.weight.device, module.weight.dtype)
    sparse_module.weight.data = module.weight.data
    if module.bias is not None:
        sparse_module.bias.data = module.bias.data
    return sparse_module

def sparsify_model(accelerator, model, train_dataloader, target_modules, args):
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if args.lora_enable:
        if args.pissa_enable:
            accelerator.print('Using PiSSA')
            peft_config = peft.LoraConfig(
                # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
                init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
                target_modules=target_modules,
                task_type=args.task_type,
            )
        else:
            accelerator.print('Using LoRA')
            peft_config = peft.LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                # bias=args.lora_bias,
                bias='none',  # bias is disabled by default
                task_type=args.task_type
            )
        model = peft.get_peft_model(model, peft_config)
        return model
    if args.sparse_enable:
        accelerator.print('Using Sparse')
        with disable_random_init_context():
            replace_layers(
                model, lambda m, n: replace_linear(m, n, target_modules, args))
        model = model.to(accelerator.device)
        update_linear_sparse(accelerator, model, train_dataloader, args)
        return model
    accelerator.print('Using Dense')
    return model