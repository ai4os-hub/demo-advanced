"""Example module of how to create and train a MNIST model using deepaas_full.
"""
import time

import api
import deepaas_full

ckpt_name = f"{time.strftime('%Y%m%d-%H%M%S')}.cp.ckpt"
model = deepaas_full.create_model()

inputs = api.config.DATA_PATH / "train-images-idx3-ubyte.gz"
target = api.config.DATA_PATH / "train-labels-idx1-ubyte.gz"
options = {"batch_size": 128, "epochs": 15, "validation_split": 0.1}
options["callbacks"] = api.utils.generate_callbacks(ckpt_name)

deepaas_full.training(model, inputs, target, **options)
model.summary()
