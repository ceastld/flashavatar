import os, sys
import random
import numpy as np
import torch
import argparse
import cv2
import time
from icecream import ic
import datetime

from scene.cameras import Camera
from scene import GaussianModel, Scene_mica, load_audio_feature
from src.deform_model import Deform_Model_audio
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams

from tqdm import tqdm
from util import *

# for train
from torch.utils.tensorboard import SummaryWriter
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep
import lpips


def img2numpy(image: torch.Tensor) -> np.ndarray:
    return (image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)


def get_save_img(img1: torch.Tensor, img2: torch.Tensor, img_res):
    img1_np = (img1 * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img2_np = (img2 * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    save_img = np.zeros((img_res, img_res * 2, 3), dtype=np.uint8)
    save_img[:, :img_res, :] = img1_np
    save_img[:, img_res:, :] = img2_np
    return save_img


class MainModel:
    def __init__(
        self,
        seed: int = 0,
        idname: str = "obama",
        image_res: int = 512,
        start_checkpoint: str = None,
        logname: str = "log",
        is_infer: bool = False,
        use_args: bool = False,
        max_train_num: int = 10000,
        iterations: int = 150000,
    ) -> None:
        # args
        self.seed = seed
        self.idname = idname
        self.image_res = image_res
        self.start_checkpoint = start_checkpoint
        self.logname = logname
        self.iterations = iterations

        self.is_infer = is_infer
        self.parse_args(use_args)
        self.batch_size = 1
        set_random_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_data(
            idname=idname,
            logname=logname,
            white_background=self.lpt.white_background,
            max_train_num=max_train_num,
        )

        self.init_model()

        if not self.start_checkpoint and is_infer:
            self.start_checkpoint = os.path.join(self.log_dir, "ckpt", "chkpnt150000.pth")

        self.first_iter = 0
        # restore
        if self.start_checkpoint:
            (model_params, gauss_params, first_iter) = torch.load(self.start_checkpoint)
            self.DeformModel.restore(model_params)
            self.gaussians.restore(gauss_params, self.opt)
            self.first_iter = first_iter

        bg_color = [1, 1, 1] if self.lpt.white_background else [0, 1, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        self.viewpoint = self.scene.getCameras().copy()
        self.codedict = {}
        self.codedict["shape"] = self.scene.shape_param.to(self.device)
        self.DeformModel.example_init(self.codedict)

    def load_data(self, idname, logname, white_background, max_train_num):
        self.data_dir = os.path.join("dataset", idname)
        self.mica_datadir = os.path.join("metrical-tracker/output", idname)
        self.log_dir = os.path.join(self.data_dir, logname)
        self.train_dir = os.path.join(self.log_dir, "train")
        self.model_dir = os.path.join(self.log_dir, "ckpt")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.doc_dir = os.path.join(self.log_dir, "doc")
        os.makedirs(self.doc_dir, exist_ok=True)

        self.scene = Scene_mica(self.data_dir, self.mica_datadir, train_type=0, white_background=white_background, device=device, max_train_num=max_train_num)

    def init_model(self):
        DeformModel = Deform_Model_audio(self.device).to(self.device)
        DeformModel.training_setup()
        if self.is_infer:
            DeformModel.eval()
        self.DeformModel = DeformModel

        gaussians = GaussianModel(self.lpt.sh_degree)
        gaussians.training_setup(self.opt)
        self.gaussians = gaussians

    def train(self):
        percep_module = lpips.LPIPS(net="vgg").to(self.device)
        writer = SummaryWriter(self.log_dir)
        first_iter = self.first_iter
        viewpoint_stack = None
        first_iter += 1

        mid_num = 30000

        codedict = self.codedict
        gaussians = self.gaussians
        DeformModel = self.DeformModel

        for iteration in range(first_iter, self.iterations + 1):
            # Every 500 its we increase the levels of SH up to a maximum degree
            if iteration % 500 == 0:
                gaussians.oneupSHdegree()

            # random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getCameras().copy()
                random.shuffle(viewpoint_stack)
                if len(viewpoint_stack) > 2000:
                    viewpoint_stack = viewpoint_stack[:2000]
            viewpoint_cam: Camera = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
            frame_id = viewpoint_cam.uid

            # deform gaussians
            codedict["expr"] = viewpoint_cam.exp_param
            codedict["eyes_pose"] = viewpoint_cam.eyes_pose
            codedict["eyelids"] = viewpoint_cam.eyelids
            codedict["jaw_pose"] = viewpoint_cam.jaw_pose
            codedict["audio_feature"] = viewpoint_cam.audio_feature

            # expr_code, eyes_pose, eyelids, jaw_pose = DeformModel.decode_audio(viewpoint_cam.audio_feature)
            # 不要回归表情系数

            verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

            if iteration == 1:
                gaussians.create_from_verts(verts_final[0])
                gaussians.training_setup(self.opt)
            gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

            # Render
            render_pkg = render(viewpoint_cam, gaussians, self.ppt, self.background)
            image = render_pkg["render"]

            # Loss
            gt_image = viewpoint_cam.original_image
            mouth_mask = viewpoint_cam.mouth_mask

            loss_huber = huber_loss(image, gt_image, 0.1) + 40 * huber_loss(image * mouth_mask, gt_image * mouth_mask, 0.1)

            loss_G = 0.0
            head_mask = viewpoint_cam.head_mask
            image_percep = normalize_for_percep(image * head_mask)
            gt_image_percep = normalize_for_percep(gt_image * head_mask)
            if iteration > mid_num:
                loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * 0.05

            loss = loss_huber * 1 + loss_G * 1

            loss.backward()

            with torch.no_grad():
                # Optimizer step
                if iteration < self.iterations:
                    gaussians.optimizer.step()
                    DeformModel.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    DeformModel.optimizer.zero_grad(set_to_none=True)

                # print loss
                if iteration % 500 == 0:
                    if iteration <= mid_num:
                        print("step: %d, huber: %.5f" % (iteration, loss_huber.item()))
                    else:
                        print("step: %d, huber: %.5f, percep: %.5f" % (iteration, loss_huber.item(), loss_G.item()))

                if iteration % 500 == 0:
                    writer.add_scalar("loss", loss.item(), iteration)

                # visualize results
                if iteration % 500 == 0 or iteration == 1:
                    save_img = get_save_img(gt_image, image.clamp(0, 1), self.image_res)
                    cv2.imwrite(os.path.join(self.log_dir, "train", f"{iteration}.jpg"), save_img[:, :, [2, 1, 0]])

                # save checkpoint
                if iteration % 5000 == 0:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (DeformModel.capture(), gaussians.capture(), iteration),
                        os.path.join(self.model_dir, f"chkpnt{iteration}.pth"),
                    )

    def infer_video(self, video_path, max_frames=100000, cam_id=0):
        video: VideoFileClip = VideoFileClip(video_path).set_fps(25)
        audio_path = extract_audio_from_video(video_path)
        audio_feature_path = extract_audio_feature(audio_path)
        audio_feature_list = load_audio_feature(audio_feature_path, self.device)
        test_path = str(Path(video_path).with_stem("test"))
        res = self.image_res
        duration = video.duration
        fps = video.fps
        total_frames = min(int(duration * fps), max_frames)

        writer = FFmpegPipeWrapper(test_path, [res * 2, res], 25)  # 1024 * 512
        for i, frame in tqdm(enumerate(video.iter_frames()), total=total_frames, desc="render video"):
            if 2 * i + 1 >= audio_feature_list.shape[0]:
                break
            if i >= max_frames:
                break
            audio_feature = audio_feature_list[2 * i : 2 * i + 16, :]
            image = self.render(self.viewpoint[cam_id], audio_feature)
            image_np = img2numpy(image)
            save_img = np.zeros((res, res * 2, 3), dtype=np.uint8)
            save_img[:, :res, :] = frame
            save_img[:, res:, :] = image_np
            writer.write_frame(save_img.tobytes())
        writer.close()
        add_audio_to_video(test_path, audio_path)

    def render(self, viewpoint_cam, audio_feature):
        self.codedict["audio_feature"] = audio_feature

        verts_final, rot_delta, scale_coef = self.DeformModel.decode(self.codedict)
        self.gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, self.gaussians, self.ppt, self.background)
        image = render_pkg["render"]
        return image

    def infer(self, totalframes=500, audio_path=None):
        audio_feature_s = extract_audio_feature(audio_path)
        test_path = os.path.join(self.log_dir, "test.mp4")
        writer = FFmpegPipeWrapper(test_path, [1024, 512], 25)
        for iteration in tqdm(range(totalframes), desc="write to video"):
            viewpoint_cam: Camera = self.viewpoint[iteration]

            if audio_path:
                audio_feature = (audio_feature_s[2 * iteration, :] + audio_feature_s[2 * iteration + 1, :]) / 2

            image = self.render(viewpoint_cam, audio_feature)
            save_img = get_save_img(viewpoint_cam.original_image, image.clamp(0, 1), self.image_res)
            writer.write_frame(save_img.tobytes())

        writer.close()
        print(f"video saved at {test_path}")

    def parse_args(self, use_args: bool):
        parser = argparse.ArgumentParser(description="Model parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--seed", type=int, default=0, help="Random seed.")
        parser.add_argument("--idname", type=str, default="obama", help="id name")
        parser.add_argument("--image_res", type=int, default=512, help="image resolution")
        parser.add_argument("--start_checkpoint", type=str, default=None)
        parser.add_argument("--logname", type=str, default="log_audio", help="log name")
        args = parser.parse_args(sys.argv[1:])
        self.lpt = lp.extract(args)
        self.opt = op.extract(args)
        self.ppt = pp.extract(args)
        if use_args:
            self.seed = args.seed
            self.idname = args.idname
            self.image_res = args.image_res
            self.start_checkpoint = args.start_checkpoint
            self.logname = args.logname
