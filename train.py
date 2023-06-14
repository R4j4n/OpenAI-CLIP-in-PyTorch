import datetime
import itertools
import logging
import os
import json

import timm
import torch
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from tqdm.autonotebook import tqdm


from config import cfg
from models import CLIPModel, ImageEncoder, ProjectionHead, TextEncoder
from utils import AvgMeter, CLIPDataset, get_lr


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{cfg.paths.new_csv}")
    max_id = dataframe["id"].max() + 1 
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe



def get_transforms(mode="train"):
    
    img_enc = ImageEncoder()

    data_config = img_enc.data_config
    
    if mode=="train":
    
        transforms = timm.data.create_transform(**data_config, is_training=True)
        return transforms
    else:
        
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return transforms


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        image_filenames = dataframe["image"].values,
        captions= dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
        image_path=cfg.paths.images_path
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(cfg.train.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(cfg.train.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


if __name__ == "__main__":
    
    # LOGGING
    current_time = datetime.datetime.now().strftime("%B%m-%d-%H-%M-%S")
    log_dir = os.path.join("log", current_time)
    os.makedirs(log_dir, exist_ok=True)
    set_logger(os.path.join(log_dir, "train.log"))
    
    logging.info("Configuration:\n" + json.dumps(cfg, indent=4, sort_keys=True))
    
    text_encoder = TextEncoder()
    tokenizer = text_encoder.tokenizer
    del text_encoder
    
    train_df, valid_df = make_train_valid_dfs()
    
    
    # train_df.to_csv(f"{log_dir}/train.csv")
    valid_df.to_csv(f"{log_dir}/valid.csv")
    
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    
    model = CLIPModel().to(cfg.train.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": cfg.train.text_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": cfg.train.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": cfg.train.head_lr, "weight_decay": cfg.train.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.train.patience, factor=cfg.train.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(cfg.train.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss.avg}")
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        logging.info(f"Epoch {epoch+1}, Valid Loss: {valid_loss.avg}")
        
        
        model_path = os.path.join(log_dir, f"Epoch-{epoch}_Model.pt")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Epoch {epoch} Model Saved with val loss: {valid_loss.avg} ")
    
        lr_scheduler.step(valid_loss.avg)
        
        
