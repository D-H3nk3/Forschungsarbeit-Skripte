import pandas as pd
from torchvision.io import read_image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, image_df):
        self.image_df = image_df
        # ToDo: Basepath immer anpassen!!!
        self.basepath = Path(
            "./Downloads/Dataset/data/RawData"
        )

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        img_path = self.basepath.joinpath(self.image_df.iloc[idx]["picture_path"])
        image = read_image(str(img_path))
        gray_img = T.Grayscale()(image)
        resized_image = T.Resize(64)(gray_img)
        label = torch.tensor(self.image_df.iloc[idx].values[1:5].astype("float32"))
        return resized_image, label

    def visualize_image(self, idx):
        figure = plt.figure(figsize=(8, 8))
        img, label = self[idx]
        plt.title(f"Picture with index {idx}")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.show()
        print(f"Diameter: {label[0]}")
        print(f"Surface tension coefficient: {label[1]}")
        print(f"Reservoir depth: {label[2]}")
        print(f"Impact velocity: {label[3]}")


class DropData:
    def __init__(self, train_portion=0.8) -> None:
        self.image_df = pd.read_csv(
            "./Downloads/Dataset/data/results.csv"
        )
        num_elements = len(self.image_df)
        num_train_elements = int(np.floor(train_portion * num_elements))
        print("Generated datasets consists of...")
        print(f"{num_train_elements} training datapoints.")
        print(f"{num_elements - num_train_elements} training datapoints.")
        print(f"Available data consists of {num_elements} data.")
        self.train_df = self.image_df.iloc[:num_train_elements]
        self.test_df = self.image_df.iloc[num_train_elements:]

    def get_train_data(self):
        return CustomImageDataset(self.train_df)

    def get_test_data(self):
        return CustomImageDataset(self.test_df)


def main():

    drop_data = DropData(train_portion=0.7)
    train_data = drop_data.get_train_data()
    print(f"Train dataset consists of {len(train_data)} elements")
    train_data.visualize_image(3)
    test_data = drop_data.get_test_data()
    print(f"Test dataset consists of {len(test_data)} elements")
    test_data.visualize_image(3)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


if __name__ == "__main__":
    main()
