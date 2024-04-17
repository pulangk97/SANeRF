


from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import gin
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tqdm import tqdm

from dataset.parsers import get_parser
from dataset.utils import io as data_io
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
from utils.tensor_dataclass import TensorDataclass
from utils.common import downsample


@gin.configurable()
class RayDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        scene: str = 'lego',
        scene_type: str = 'nerf_synthetic_multiscale',
        split: str = 'train',
        num_rays: int = 8192,
        render_bkgd: str = "white", #'white' 
        num_views: int = None,
        down_scale: int = 1,
        **kwargs
    ):
        super().__init__()

        parser = get_parser(scene_type)
        data_source = parser(
            base_path=Path(base_path), scene=scene, split=split, down_scale = down_scale, **kwargs
        )
        self.training = split.find('train') >= 0



        self.cameras = data_source['cameras']
        self.ray_bundles = [c.build('cpu') for c in self.cameras]
        logger.info('==> Find {} cameras'.format(len(self.cameras)))
        self.poses = {
            k: torch.tensor(np.asarray(v)).float()  # Nx4x4
            for k, v in data_source["poses"].items()
        }



        # parallel loading frames
        self.frames = {}
        for k, cam_frames in data_source['frames'].items():
            with ThreadPoolExecutor(
                max_workers=min(multiprocessing.cpu_count(), 32)
            ) as executor:
                frames = list(
                    tqdm(
                        executor.map(
                            lambda f: torch.tensor(
                                data_io.imread(f['image_filename'])
                            ),
                            cam_frames,
                        ),
                        total=len(cam_frames),
                        dynamic_ncols=True,
                    )
                )
            self.frames[k] = torch.stack(frames, dim=0)

        self.aabb = torch.tensor(np.asarray(data_source['aabb'])).float()
        self.loss_multi = {
            k: torch.tensor([x['lossmult'] for x in v])
            for k, v in data_source['frames'].items()
        }
        self.file_names = {
            k: [x['image_filename'].stem for x in v]
            for k, v in data_source['frames'].items()
        }
        self.num_rays = num_rays
        self.render_bkgd = render_bkgd

        len_images = self.frames[0].shape[0] 


        ##  Adhere to the split methodology defined in DietNeRF (https://github.com/ajayjain/DietNeRF)
        ##  cite: Ajay Jain, Matthew Tancik, and Pieter Abbeel. 2021. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5885–5894.
        if num_views!=None:

            split_indices = {
                'test': [i for i in np.arange(0,len_images,np.int(len_images/25))],
                'train': [26,86,2,55,75,93,16,73,8],
            }

            indices = split_indices[split]
            if split=="train":
                num_views = num_views
            else:
                num_views = 200
            for i in range(len(self.frames)):
                
                self.frames[i] = self.frames[i][indices][:num_views]
                self.loss_multi[i] = self.loss_multi[i][indices][:num_views]
                self.poses[i] = self.poses[i][indices][:num_views]

                selected_filenames = [self.file_names[i][j] for j in indices]   
                self.file_names[i] = selected_filenames[:num_views]

        if down_scale>0:
            for i in range(len(self.frames)):
                images = []
                for j in range(self.frames[0].shape[0]):
                    images.append(torch.tensor(downsample((self.frames[i][j]).numpy(), down_scale)))
                images = torch.stack(images)
                self.frames[i] = images
            print("dowm sampled image size: "+str(self.frames[0].shape))
        

        ## Convert to world coordinates early 
        for i in range(len(self.poses)):
            directions = []
            origins = []
            radiis = []
            ray_cos = []
            for j in range(len(self.poses[i])):
                directions.append( (
                    self.poses[i][j][:3, :3] @ self.ray_bundles[i].directions[..., None]
                ).squeeze(-1))
                origins.append(torch.broadcast_to(self.poses[i][j][:3, -1],(self.ray_bundles[i].radiis.shape[0],self.ray_bundles[i].radiis.shape[1],-1)))
                radiis.append(self.ray_bundles[i].radiis)
                ray_cos.append(self.ray_bundles[i].ray_cos)
            directions = torch.stack(directions,dim=0)
            origins = torch.stack(origins,dim=0)
            radiis = torch.stack(radiis,dim=0)
            ray_cos = torch.stack(ray_cos,dim=0)
     
            self.ray_bundles[i] = RayBundle(
                origins=origins.squeeze(),
                directions=directions.squeeze(),
                radiis=radiis,
                ray_cos=ray_cos)

        self.split = split
 


        self.frame_number = {k: x.shape[0] for k, x in self.frames.items()}


        # try to read a data to initialize RenderBuffer subclass
        self[0]

    def __len__(self):
        if self.training:
            return 10**9  # hack of streaming dataset
        else:
            return sum([x.shape[0] for k, x in self.poses.items()])

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    @torch.no_grad()
    def __getitem__(self, index):
        if self.training:
            rgb, c2w, cam_rays, loss_multi = [], [], [], []
            for cam_idx in range(len(self.cameras)):
                num_rays = int(
                    self.num_rays
                    * (1.0 / self.loss_multi[cam_idx][0])
                    / sum([1.0 / v[0] for _, v in self.loss_multi.items()])
                )
                idx = torch.randint(
                    0,
                    self.frames[cam_idx].shape[0],
                    size=(num_rays,),
                )
                sample_x = torch.randint(
                    0,
                    self.cameras[cam_idx].width,
                    size=(num_rays,),
                )  # uniform sampling
                sample_y = torch.randint(
                    0,
                    self.cameras[cam_idx].height,
                    size=(num_rays,),
                )  # uniform sampling
                rgb.append(self.frames[cam_idx][idx, sample_y, sample_x])
                c2w.append(self.poses[cam_idx][idx])
                cam_rays.append(self.ray_bundles[cam_idx][idx, sample_y, sample_x])
                loss_multi.append(self.loss_multi[cam_idx][idx, None])


            rgb = torch.cat(rgb, dim=0)
            c2w = torch.cat(c2w, dim=0)
            cam_rays = RayBundle.direct_cat(cam_rays, dim=0)
            loss_multi = torch.cat(loss_multi, dim=0)

            if 'white' == self.render_bkgd:
                render_bkgd = torch.ones_like(rgb[..., [-1]])
            elif 'rand' == self.render_bkgd:
                render_bkgd = torch.rand_like(rgb[..., :3])
            elif 'randn' == self.render_bkgd:
                render_bkgd = (torch.randn_like(rgb[..., :3]) + 0.5).clamp(
                    0.0, 1.0
                )
            else:
                raise NotImplementedError

        else:
            for cam_idx, num in self.frame_number.items():
                if index < num:
                    break
                index = index - num
            num_rays = 1
            idx = torch.ones(size=(num_rays,), dtype=torch.int64) * index
            sample_x, sample_y = torch.meshgrid(
                torch.arange(self.cameras[cam_idx].width),
                torch.arange(self.cameras[cam_idx].height),
                indexing="xy",
            )
            sample_x = sample_x.reshape(-1)
            sample_y = sample_y.reshape(-1)

            rgb = self.frames[cam_idx][idx, sample_y, sample_x]
            c2w = self.poses[cam_idx][idx]
            cam_rays = self.ray_bundles[cam_idx][idx, sample_y, sample_x]
            loss_multi = self.loss_multi[cam_idx][idx, None]
            render_bkgd = torch.ones_like(rgb[..., [-1]])

        target = RenderBuffer(
            rgb=rgb[..., :3] * rgb[..., [-1]]
            + (1.0 - rgb[..., [-1]]) * render_bkgd,
            render_bkgd=render_bkgd,
            loss_multi=loss_multi,
        )

        if not self.training:
            cam_rays = cam_rays.reshape(
                (self.cameras[cam_idx].height, self.cameras[cam_idx].width)
            )
            target = target.reshape(
                (self.cameras[cam_idx].height, self.cameras[cam_idx].width)
            )
        outputs = {
            # 'c2w': c2w,
            'cam_rays': cam_rays,
            'target': target,
            # 'idx': idx,
        }
        if not self.training:
            outputs['name'] = self.file_names[cam_idx][index]
        return outputs



def ray_collate(batch):
    res = {k: [] for k in batch[0].keys()}
    for data in batch:
        for k, v in data.items():
            res[k].append(v)
    for k, v in res.items():
        if isinstance(v[0], RenderBuffer) or isinstance(v[0], RayBundle):
            res[k] = TensorDataclass.direct_cat(v, dim=0)
        else:
            res[k] = torch.cat(v, dim=0)
    return res


if __name__ == '__main__':
    training_dataset = RayDataset(
        # '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic',
        '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic_multiscale',
        'lego',
        # 'nerf_synthetic',
        'nerf_synthetic_multiscale',
    )
    train_loader = iter(
        DataLoader(
            training_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=ray_collate,
            pin_memory=True,
            worker_init_fn=None,
            pin_memory_device='cuda',
        )
    )
    test_dataset = RayDataset(
        # '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic',
        '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic_multiscale',
        'lego',
        # 'nerf_synthetic',
        'nerf_synthetic_multiscale',
        num_rays=81920,
        split='test',
    )
    for i in tqdm(range(1000)):
        data = next(train_loader)
        pass
    for i in tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        pass
    pass
