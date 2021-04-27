import math
from pathlib import Path
from typing import List, Dict, Any

import torch
import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from dp.dataset import new_dataloader
from dp.decorators import ignore_exception
from dp.losses import CrossEntropyLoss, CTCLoss
from dp.metrics import phoneme_error_rate, word_error
from dp.model import Model
from dp.predictor import Predictor
from dp.text import Preprocessor
from dp.utils import to_device, unpickle_binary


class Trainer:

    def __init__(self, checkpoint_dir: str, loss_type='ctc') -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir / 'tensorboard')
        self.loss_type = loss_type
        if loss_type == 'ctc':
            self.criterion = CTCLoss()
        elif loss_type == 'cross_entropy':
            self.criterion = CrossEntropyLoss()
        else:
            raise ValueError(f'Loss not supported: {loss_type}')

    def train(self,
              model: Model,
              checkpoint: dict,
              store_phoneme_dict_in_model=True) -> None:

        config = checkpoint['config']
        data_dir = Path(config['paths']['data_dir'])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        model.train()

        criterion = self.criterion.to(device)

        optimizer = Adam(model.parameters())
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = config['training']['learning_rate']

        train_loader = new_dataloader(dataset_file=data_dir / 'train_dataset.pkl',
                                      drop_last=True, batch_size=config['training']['batch_size'])
        val_loader = new_dataloader(dataset_file=data_dir / 'val_dataset.pkl',
                                    drop_last=False, batch_size=config['training']['batch_size_val'])
        if store_phoneme_dict_in_model:
            phoneme_dict = unpickle_binary(data_dir / 'phoneme_dict.pkl')
            checkpoint['phoneme_dict'] = phoneme_dict

        val_batches = sorted([b for b in val_loader], key=lambda x: -x['text_len'][0])

        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=config['training']['scheduler_plateau_factor'],
                                      patience=config['training']['scheduler_plateau_patience'],
                                      mode='min')
        loss_sum = 0.
        best_per = math.inf        
        if 'step' not in checkpoint:
            checkpoint['step'] = 0
        start_epoch = checkpoint['step'] // len(train_loader)

        for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
            pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
            for i, batch in pbar:
                checkpoint['step'] += 1
                step = checkpoint['step']

                self._set_warmup_lr(optimizer=optimizer, step=step,
                                    config=config)
                batch = to_device(batch, device)
                pbar.set_description(desc=f'Epoch: {epoch} | Step {step} '
                                          f'| Loss: {loss_sum / i:#.4}', refresh=True)

                pred = model(batch)
                loss = criterion(pred, batch)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_sum += loss.item()

                self.writer.add_scalar('Loss/train', loss.item(), global_step=step)
                self.writer.add_scalar('Params/batch_size', config['training']['batch_size'],
                                       global_step=step)
                self.writer.add_scalar('Params/learning_rate', [g['lr'] for g in optimizer.param_groups][0],
                                       global_step=step)

                if step % config['training']['validate_steps'] == 0:
                    val_loss = self.validate(model, val_batches)
                    self.writer.add_scalar('Loss/val', val_loss, global_step=step)

                if step % config['training']['generate_steps'] == 0:
                    per = self.generate_samples(model=model,
                                                preprocessor=checkpoint['preprocessor'],
                                                val_batches=val_batches,
                                                n_log_samples=config['training']['n_generate_samples'],
                                                step=step)
                    if per is not None and per < best_per:
                        self.save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                        path=self.checkpoint_dir / f'best_model.pt')
                        self.save_model(model=model, optimizer=None, checkpoint=checkpoint,
                                        path=self.checkpoint_dir / f'best_model_no_optim.pt')
                        scheduler.step(per)

                if step % config['training']['checkpoint_steps'] == 0:
                    step = step // 1000
                    self.save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                    path=self.checkpoint_dir / f'model_step_{step}k.pt')

            loss_sum = 0
            self.save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                            path=self.checkpoint_dir / 'latest_model.pt')

    def validate(self, model: Model, val_batches: List[dict]) -> float:
        device = next(model.parameters()).device
        criterion = self.criterion.to(device)
        model.eval()
        val_losses = []
        for batch in val_batches:
            batch = to_device(batch, device)
            with torch.no_grad():
                pred = model(batch)
                loss = criterion(pred, batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())
        model.train()
        return sum(val_losses) / len(val_losses)

    @ignore_exception
    def generate_samples(self,
                         model: Model,
                         preprocessor: Preprocessor,
                         val_batches: List[dict],
                         n_log_samples: int,
                         step: int) -> float:
        """ Generates samples and calculates some metrics. Returns phoneme error rate. """

        device = next(model.parameters()).device
        model.eval()
        text_tokenizer = preprocessor.text_tokenizer
        phoneme_tokenizer = preprocessor.phoneme_tokenizer
        lang_tokenizer = preprocessor.lang_tokenizer
        lang_prediction_result = dict()

        for batch in val_batches:
            batch = to_device(batch, device)
            batch['start_index'] = batch['phonemes'][:, 0]
            generated_batch, _ = model.generate(batch)
            for i in range(batch['text'].size(0)):
                text_len = batch['text_len'][i]
                text = batch['text'][i, :text_len]
                target = batch['phonemes'][i, :]
                lang = batch['language'][i]
                lang = lang_tokenizer.decode(lang.detach().cpu().item())
                generated = generated_batch[i, :text_len].cpu()
                text, target = text.detach().cpu(), target.detach().cpu()
                text = text_tokenizer.decode(text, remove_special_tokens=True)
                generated = phoneme_tokenizer.decode(generated, remove_special_tokens=True)
                target = phoneme_tokenizer.decode(target, remove_special_tokens=True)
                lang_prediction_result[lang] = lang_prediction_result.get(lang, []) + [(text, generated, target)]

        # calculate error rates per language
        lang_per, lang_wer = dict(), dict()
        languages = sorted(lang_prediction_result.keys())
        for lang in languages:
            log_texts = []
            for text, generated, target in lang_prediction_result[lang]:
                per = phoneme_error_rate(generated, target)
                wer = word_error(generated, target)
                lang_per[lang] = lang_per.get(lang, []) + [per]
                lang_wer[lang] = lang_wer.get(lang, []) + [wer]
                text, gen_decoded, target = ''.join(text), ''.join(generated), ''.join(target)
                log_texts.append(f'     {text:<30} {gen_decoded:<30} {target:<30}')

            self.writer.add_text(f'Text_Prediction_Target/{lang}',
                                 '\n'.join(log_texts[:n_log_samples]), global_step=step)

        sum_wer, sum_per, count = 0., 0., 0
        for lang in languages:
            count += len(lang_per[lang])
            sum_per = sum_per + sum(lang_per[lang])
            sum_wer = sum_wer + sum(lang_wer[lang])
            per = sum(lang_per[lang]) / len(lang_per[lang])
            wer = sum(lang_wer[lang]) / len(lang_wer[lang])
            self.writer.add_scalar(f'Phoneme_Error_Rate/{lang}', per, global_step=step)
            self.writer.add_scalar(f'Word_Error_Rate/{lang}', wer, global_step=step)
        self.writer.add_scalar(f'Phoneme_Error_Rate/mean', sum_per / count, global_step=step)
        self.writer.add_scalar(f'Word_Error_Rate/mean', sum_wer / count, global_step=step)

        model.train()

        return sum_per / count

    def save_model(self,
                   model: torch.nn.Module,
                   optimizer: torch.optim,
                   checkpoint: Dict[str, Any],
                   path: Path) -> None:
        checkpoint['model'] = model.state_dict()
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        else:
            checkpoint['optimizer'] = None
        torch.save(checkpoint, str(path))

    def _set_warmup_lr(self,
                       optimizer: torch.optim,
                       step: int,
                       config: Dict[str, Any]) -> None:

        warmup_steps = config['training']['warmup_steps']
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = 1.0 - max(warmup_steps - step, 0) / warmup_steps
            for g in optimizer.param_groups:
                g['lr'] = config['training']['learning_rate'] * warmup_factor
