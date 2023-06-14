import torch
from PIL import Image
from config import cfg
from transformers import AutoTokenizer

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]




class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self,image_path,image_filenames, captions, tokenizer, transforms):

        self.image_pth = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        
        self.tokenizer = tokenizer
        self.encoded_captions = self.tokenizer(
            
            list(captions), padding=True, truncation=True, return_tensors='pt',
            max_length=cfg.train.max_length
        )
        self.transforms = transforms
    def __getitem__(self, idx):
 
        item = {
            key: values[idx].clone().detach()
            for key, values in self.encoded_captions.items()
                }

        img = Image.open(f"{self.image_pth}/{self.image_filenames[idx]}")
        image = self.transforms(img)
        item['image'] = image
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)




