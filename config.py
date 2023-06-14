import os 
from easydict import EasyDict as edict

root_path = os.path.dirname(os.path.realpath(__file__))
    
c = edict()

# paths 
c.paths = edict()
c.paths.images_path = f'{root_path}/flickr-image-dataset/flickr30k_images/flickr30k_images'
c.paths.new_csv = f"{root_path}/flickr-image-dataset/flickr30k_images/captions.csv"

# dataloder 
c.dataloader = edict()
c.dataloader.batch_size = 32
c.dataloader.num_workers = 16

# Train
c.train = edict()
c.train.head_lr = 1e-3
c.train.image_encoder_lr = 1e-4
c.train.text_encoder_lr = 1e-5
c.train.weight_decay = 1e-3
c.train.patience = 1
c.train.factor = 0.8
c.train.epochs = 5
c.train.device = "cuda:1" 
c.train.max_length = 100


# projection 
c.projection = edict()
c.projection.image_embedding = 1000
c.projection.text_embedding = 768
c.projection.num_projection_layers = 1
c.projection.projection_dim = 512 
c.projection.dropout = 0.1


# clip
c.clip = edict()
c.clip.temperature = 1.0



cfg = c
