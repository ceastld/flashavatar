from model import MainModel

if __name__ == "__main__":
    model = MainModel(idname="obama", logname="log_audio", is_infer=True)
    # model.infer(audio_path="../video/obama28/audio.wav")
    model.infer_video("../video/obama29/video.mp4")
