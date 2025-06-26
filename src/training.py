import torch
import os
from typing import IO, BinaryIO
import json
from tqdm import tqdm
import numpy as np
from .model import TransformerLM
from .nn_tools import cross_entropy, clip_gradients
from .optimizer import AdamW, CosineWithWarmup


def train(train_data_path, val_data_path, config_path: str, checkpoint=None):
    with open(config_path) as f:
        config = json.load(f)

    max_iters = config['training']['max_iters']
    batch_size = config['data']['batch_size']
    device = config['data']['device']
    context_length = config['model']['context_length']
    eval_interval = config['training']['eval_interval']
    grad_clip = config['training']['grad_clip']
    save_interval = config['training']['save_interval']
    checkpoint_path = config['training']['checkpoint_path']

    train_data = np.load(train_data_path, mmap_mode='r')
    val_data = np.load(val_data_path, mmap_mode='r')

    model = TransformerLM(**config['model']).to(device)

    optimizer = AdamW(model.parameters(), **config['optimizer'])
    scheduler = CosineWithWarmup(optimizer, **config['scheduler'])
    val_loss = torch.tensor([0.0])
    accuracy = 0.0

    if checkpoint is not None:
        start_iter = load_checkpoint(checkpoint, model, optimizer)
    else:
        start_iter = 0

    pbar = tqdm(range(start_iter, max_iters))

    model.train()
    for iteration in pbar:
        inputs, targets = get_batch(train_data, batch_size, context_length, device)

        logits = model(inputs)

        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()

        loss.backward()

        clip_gradients(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        scheduler.step()

        pbar.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.4e}, train_loss={loss.item():.4f}, val_accuracy={accuracy:.2f}')

        if iteration % eval_interval == 0:
            model.eval()
            inputs, targets = get_batch(val_data, batch_size, context_length, device)
            with torch.no_grad():
                val_logits = model(inputs)
                val_loss = cross_entropy(val_logits, targets).item()

                pred_ids = val_logits.max(-1)[1]
                accuracy = (pred_ids == targets).float().mean().item()

            pbar.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.4e}, train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_accuracy={accuracy:.2f}')

            model.train()

        if iteration % save_interval == 0:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

    save_checkpoint(model, optimizer, iteration, checkpoint_path)


def get_batch(ids: np.ndarray, batch_size: int, context_len: int, device: str = 'cpu'):
    device = torch.device(device)

    start_ids = np.random.randint(0, len(ids) - context_len, batch_size)

    inputs = torch.stack(
        [torch.from_numpy(ids[i:i + context_len].copy()) for i in start_ids]
    )

    targets = torch.stack(
        [torch.from_numpy(ids[i + 1:i + context_len + 1].copy()) for i in start_ids]
    )

    return inputs.to(torch.long), targets.to(torch.long).to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out, pickle_protocol=5)


def save_model(
    model: torch.nn.Module,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    torch.save(model.state_dict(), out, pickle_protocol=5)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
):
    checkpoint = torch.load(src, weights_only=False)

    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint.get('iteration', 0)