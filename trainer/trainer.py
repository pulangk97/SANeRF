import time

import gin
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List

from neural_field.model.RFModel import RFModel
from utils.writer import TensorboardWriter
from utils.colormaps import apply_depth_colormap
import dataset.utils.io as data_io


@gin.configurable()
class Trainer:
    def __init__(
        self,
        model: RFModel,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        # configurable
        base_exp_dir: str = 'experiments',
        exp_name: str = 'Tri-MipRF',
        max_steps: int = 50000,
        log_step: int = 500,
        eval_step: int = 500,
        target_sample_batch_size: int = 65536,
        test_chunk_size: int = 8192,
        dynamic_batch_size: bool = True,
        num_rays: int = 8192,
        varied_eval_img: bool = True,

        ## sanerf setting
        if_anneal: bool = False,
        anneal_fs: float = 0.15,
        num_split: int = 30,
        down_speed: float = 0.2,
        total_reg_iter: int = 40000,
        if_sh_mask: bool = False,
        sh_band: int = 2


    ):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.max_steps = max_steps
        self.target_sample_batch_size = target_sample_batch_size
        # exp_dir
        self.exp_dir = Path(base_exp_dir) / exp_name
        self.log_step = log_step
        self.eval_step = eval_step
        self.test_chunk_size = test_chunk_size
        self.dynamic_batch_size = dynamic_batch_size
        self.num_rays = num_rays
        self.varied_eval_img = varied_eval_img

        ## sanerf setting
        self.if_anneal = if_anneal
        self.anneal_fs = anneal_fs
        self.num_split = num_split
        self.down_speed = down_speed
        self.total_reg_iter = total_reg_iter
        self.if_sh_mask = if_sh_mask
        self.sh_band = sh_band

        self.writer = TensorboardWriter(log_dir=self.exp_dir)

        self.optimizer = self.model.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

        # Save configure
        conf = gin.operative_config_str()
        logger.info(conf)
        self.save_config(conf)


    @staticmethod
    def compute_blur_ball(current_iter, total_reg_iter, anneal_fs, quanty_L, down_speed=2):
        if current_iter<total_reg_iter:
            x = np.float(np.int(current_iter*quanty_L/total_reg_iter))
            blur_ball = anneal_fs/(2**(down_speed*x))
        else:
            blur_ball = 0
        return blur_ball


    @staticmethod
    def compute_sh_mask(band):
        num_params = np.stack([2*i+1 for i in range(3)])
        mask = np.zeros(np.sum(num_params)).astype(int)
        band_params = np.sum(np.stack([2*i+1 for i in range(band)])).astype(int)
        mask[:band_params] = 1
        return mask


    def train_iter(self, step: int, data: Dict, logging=False):
        tic = time.time()
        cam_rays = data['cam_rays']
        num_rays = min(self.num_rays, len(cam_rays))
        cam_rays = cam_rays[:num_rays].cuda(non_blocking=True)
        target = data['target'][:num_rays].cuda(non_blocking=True)


        ## --------  Compute the size of blur ball ------------- ##
        if self.if_anneal == True:
            blur_ball = self.compute_blur_ball(step, self.total_reg_iter, self.anneal_fs, self.num_split, self.down_speed)
        else:
            blur_ball = 0

        if self.if_sh_mask == True:
            view_mask = torch.tensor(self.compute_sh_mask(self.sh_band)).cuda(non_blocking=True)
        else:
            view_mask = torch.tensor(self.compute_sh_mask(4)).cuda(non_blocking=True)



        rb = self.model(cam_rays, target.render_bkgd, blur_ball = blur_ball, view_mask = view_mask)

        loss_dict = self.model.compute_loss(cam_rays, rb, target)
        metrics = self.model.compute_metrics(cam_rays, rb, target)
        if 0 == metrics.get("rendering_samples_actual", -1):
            return metrics

        # update
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss_dict['total_loss']).backward()
        self.optimizer.step()

        # logging
        if logging:
            with torch.no_grad():
                iter_time = time.time() - tic
                remaining_time = (self.max_steps - step) * iter_time
                status = {
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'step': step,
                    'iter_time': iter_time,
                    'ETA': remaining_time,
                }
                self.writer.write_scalar_dicts(
                    ['loss', 'metrics', 'status'],
                    [
                        {k: v.item() for k, v in loss_dict.items()},
                        metrics,
                        status,
                    ],
                    step,
                )
        return metrics

    def fit(self):
        logger.info("==> Start training ...")

        iter_train_loader = iter(self.train_loader)
        iter_eval_loader = iter(self.eval_loader)
        eval_0 = next(iter_eval_loader)
        self.model.train()
        for step in range(self.max_steps):
            self.model.before_iter(step)
            metrics = self.train_iter(
                step,
                data=next(iter_train_loader),
                logging=(step % self.log_step == 0 and step > 0)
                or (step == 100),
            )
            if 0 == metrics.get("rendering_samples_actual", -1):
                continue

            self.scheduler.step()
            if self.dynamic_batch_size:
                rendering_samples_actual = metrics.get(
                    "rendering_samples_actual",
                    self.target_sample_batch_size,
                )
                self.num_rays = (
                    self.num_rays
                    * self.target_sample_batch_size
                    // rendering_samples_actual
                    + 1
                )

            self.model.after_iter(step)

            if step > 0 and step % self.eval_step == 0:
                self.model.eval()
                metrics, final_rb, target = self.eval_img(
                    next(iter_eval_loader) if self.varied_eval_img else eval_0, step,
                    compute_metrics=True,
                )
                self.writer.write_scalar_dicts(['eval'], [metrics], step)
                self.writer.write_image('eval/rgb', final_rb.rgb, step)
                self.writer.write_image('gt/rgb', target.rgb, step)
                self.writer.write_image(
                    'eval/depth',
                    apply_depth_colormap(final_rb.depth),
                    step,
                )
                self.writer.write_image('eval/alpha', final_rb.alpha, step)

                self.model.train()

        logger.info('==> Training done!')
        self.save_ckpt()

    @torch.no_grad()
    def eval_img(self, data, step, compute_metrics=True,if_eval=False):
        cam_rays = data['cam_rays'].cuda(non_blocking=True)
        target = data['target'].cuda(non_blocking=True)

        ## --------  Compute the size of blur ball （eval）------------- ##
        if self.if_anneal == True:
            blur_ball = self.compute_blur_ball(step, self.total_reg_iter, self.anneal_fs, self.num_split, self.down_speed)
        else:
            blur_ball = 0

        if self.if_sh_mask == True:
            view_mask = torch.tensor(self.compute_sh_mask(self.sh_band)).cuda(non_blocking=True)
        else:
            view_mask = torch.tensor(self.compute_sh_mask(4)).cuda(non_blocking=True)



        final_rb = None
        flatten_rays = cam_rays.reshape(-1)
        flatten_target = target.reshape(-1)
        for i in range(0, len(cam_rays), self.test_chunk_size):
            rb = self.model(
                flatten_rays[i : i + self.test_chunk_size],
                flatten_target[i : i + self.test_chunk_size].render_bkgd, blur_ball=blur_ball, view_mask = view_mask
            )
            final_rb = rb if final_rb is None else final_rb.cat(rb)
        final_rb = final_rb.reshape(cam_rays.shape)
        metrics = None
        if compute_metrics:
            metrics = self.model.compute_metrics(cam_rays, final_rb, target,if_eval)
        return metrics, final_rb, target

    @torch.no_grad()
    def eval(
        self,
        save_results: bool = False,
        rendering_channels: List[str] = ["rgb"],
    ):
        # ipdb.set_trace()
        logger.info("==> Start evaluation on testset ...")
        if save_results:
            res_dir = self.exp_dir / 'rendering'
            res_dir.mkdir(parents=True, exist_ok=True)
            results = {"names": []}
            results.update({k: [] for k in rendering_channels})

        self.model.eval()
        metrics = []
        for idx, data in enumerate(tqdm(self.eval_loader)):
            metric, rb, target = self.eval_img(data,step=self.max_steps,if_eval=True)
            metrics.append(metric)
            if save_results:
                results["names"].append(data['name'])
                for channel in rendering_channels:
                    if hasattr(rb, channel):
                        values = getattr(rb, channel).cpu().numpy()
                        if 'depth' == channel:
                            values = (values * 10000.0).astype(
                                np.uint16
                            )  # scale the depth by 10k, and save it as uint16 png images
                        results[channel].append(values)
                    else:
                        raise NotImplementedError
            del rb
        if save_results:
            for idx, name in enumerate(tqdm(results['names'])):
                for channel in rendering_channels:
                    channel_path = res_dir / channel
                    data = results[channel][idx]
                    data_io.write_rendering(data, channel_path, name)

        metrics = {k: [dct[k] for dct in metrics] for k in metrics[0]}
        logger.info("==> Evaluation done")
        for k, v in metrics.items():
            metrics[k] = sum(v) / len(v)
        self.writer.write_scalar_dicts(['benchmark'], [metrics], 0)
        self.writer.tb_writer.close()




    def save_config(self, config):
        dest = self.exp_dir / 'config.gin'
        if dest.exists():
            return
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / 'config.gin', 'w') as f:
            f.write(config)
        md_config_str = gin.config.markdown(config)
        self.writer.write_config(md_config_str)
        self.writer.tb_writer.flush()

    def save_ckpt(self):
        dest = self.exp_dir / 'model.ckpt'
        logger.info('==> Saving checkpoints to ' + str(dest))
        torch.save(
            {
                "model": self.model.state_dict(),
            },
            dest,
        )

    def load_ckpt(self):
        dest = self.exp_dir / 'model.ckpt'
        loaded_state = torch.load(dest, map_location="cpu")
        logger.info('==> Loading checkpoints from ' + str(dest))
        self.model.load_state_dict(loaded_state['model'])

    def load_ckpt_frompath(self,path):
        # dest = self.exp_dir / 'model.ckpt'
        loaded_state = torch.load(path, map_location="cpu")
        logger.info('==> Loading checkpoints from ' + str(path))
        self.model.load_state_dict(loaded_state['model'])

    @gin.configurable()
    def get_scheduler(self, gamma=0.6, **kwargs):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                self.max_steps // 2,
                self.max_steps * 3 // 4,
                self.max_steps * 5 // 6,
                self.max_steps * 9 // 10,
            ],
            gamma=gamma,
            **kwargs,
        )
        return scheduler
