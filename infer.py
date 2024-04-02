from model import MainModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = MainModel(
    idname="obama79",
    logname="log3",
    is_infer=True,
    start_checkpoint="dataset/obama79/log3/ckpt/chkpnt150000.pth",
    max_train_num=20,
)
# model.infer(audio_path="../video/obama28/audio.wav")
model.infer_video("../video/obama74/video.mp4", max_frames=500, cam_id=1)
