import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov
from icecream import ic
from torchvision import transforms, datasets
import torchvision.io as torchio


class Scene_loader:
    def __init__(self) -> None:

        pass


def load_audio_feature(filename: str, device):
    """_summary_

    Args:
        filename (str): _description_
        device (_type_): _description_

    Returns:
        Tensor: n*768
    """
    np_audio = np.load(filename)
    audio_feature_s = torch.from_numpy(np_audio).to(device)  # n * 768
    feature_dim = audio_feature_s.shape[1]
    blank_feature = torch.zeros((8, feature_dim)).to(device)

    audio_feature_s = torch.cat(
        (
            blank_feature,
            audio_feature_s,
            blank_feature,
        ),
        dim=0,
    )  # 后面要取16帧的窗口，所以需要在两边添加空白列
    return audio_feature_s


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device, max_train_num=10000):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1  # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = f"{datadir}_pm/seg_masks"

        self.bg_image = torch.zeros((3, 512, 512)).to(device)
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, "checkpoint")
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500
        eval_num = 50
        train_num = min(max_train_num, self.N_frames)
        ckpt_path = os.path.join(mica_ckpt_dir, "00000.frame")
        payload = torch.load(ckpt_path)
        flame_params = payload["flame"]
        self.shape_param = torch.as_tensor(flame_params["shape"])
        orig_w, orig_h = payload["img_size"]
        K = payload["opencv"]["K"][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 0
            range_up = train_num
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames

        audio_feature_file = os.path.join(datadir, "audio_feature.npy")
        audio_feature_s = load_audio_feature(audio_feature_file, device)

        # torchio.read_image(image_path)

        transform = transforms.ToTensor()

        def load_image(image_path):
            image = Image.open(image_path)
            return transform(image).to(device)

        # datasets.ImageFolder(root=alpha_folder,)

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5)  # obey mica tracking
            image_name_ori = str(frame_id + frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica + ".frame")
            payload = torch.load(ckpt_path)

            flame_params = payload["flame"]
            exp_param = torch.as_tensor(flame_params["exp"])  # 100
            eyes_pose = torch.as_tensor(flame_params["eyes"])  # 12
            eyelids = torch.as_tensor(flame_params["eyelids"])  # 2
            jaw_pose = torch.as_tensor(flame_params["jaw"])  # 6
            # sum = 120

            oepncv = payload["opencv"]
            w2cR = oepncv["R"][0]
            w2cT = oepncv["t"][0]
            R = np.transpose(w2cR)  # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(images_folder, image_name_ori + ".jpg")
            resized_image_rgb = load_image(image_path)
            gt_image = resized_image_rgb[:3, ...]

            # seg_mask
            alpha_path = os.path.join(alpha_folder, str(frame_id + frame_delta).zfill(6) + ".png")
            alpha = load_image(alpha_path)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori + "_neckhead.png")
            head_mask = load_image(head_mask_path)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori + "_mouth.png")
            mouth_mask = load_image(mouth_mask_path)

            # audio feature
            # audio_feature = (audio_feature_s[2 * frame_id, :] + audio_feature_s[2 * frame_id + 1, :]) / 2
            af_id = 2 * frame_id
            audio_feature = audio_feature_s[af_id : af_id + 16, :]  # 16 * 768

            camera_indiv = Camera(
                colmap_id=frame_id,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image=gt_image.clamp(0, 1),
                head_mask=head_mask,
                mouth_mask=mouth_mask,
                exp_param=exp_param,
                eyes_pose=eyes_pose,
                eyelids=eyelids,
                jaw_pose=jaw_pose,
                audio_feature=audio_feature,  # ldy
                image_name=image_name_mica,
                uid=frame_id,
                data_device=device,
            )
            self.cameras.append(camera_indiv)

    def getCameras(self):
        return self.cameras
