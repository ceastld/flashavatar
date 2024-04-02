from model import MainModel
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = MainModel(
    idname="obama79",
    logname="log4",
    max_train_num=1000,
)

model.train()
