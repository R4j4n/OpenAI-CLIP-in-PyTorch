"""

Download the data and process it. 
Download the kaggle api key json and place it inside the directory.

"""

import os
import pandas as pd
import opendatasets as od

Kaggle_URL = "https://www.kaggle.com/datasets/adityajn105/flickr30k"
root_path = os.path.dirname(os.path.realpath(__file__))

# download the data
od.download(Kaggle_URL)


df = pd.read_csv(
    f"{root_path}/flickr-image-dataset/flickr30k_images/results.csv", delimiter="|"
)
df.columns = ["image", "caption_number", "caption"]
df["caption"] = df["caption"].str.lstrip()
df["caption_number"] = df["caption_number"].str.lstrip()
df.loc[19999, "caption_number"] = "4"
df.loc[19999, "caption"] = "A dog runs across the grass ."
ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
df["id"] = ids
df.to_csv(
    f"{root_path}/flickr-image-dataset/flickr30k_images/captions.csv", index=False
)
df.head()

print("Data Download Sucessful........")
