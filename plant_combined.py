import torch
import torch.nn as nn
from plant_ann import PlantANN
from plant_cnn import PlantCNN


class Plant_Combined(nn.Module):
    def __init__(self):
        super(Plant_Combined, self).__init__()
        self.ann = PlantANN()
        self.cnn = PlantCNN()
        self.network = nn.Sequential(
            nn.Linear(15,10),
            nn.LeakyReLU(),
            nn.Linear(10,2)
        )

    def forward(self, data, image):
        data = self.ann(data)
        image = self.cnn(image)
        x = torch.hstack((data, image))
        return self.network(x)


if __name__ == "__main__":
    random_tensor = torch.rand((100,52), dtype=torch.float32)
    model = PlantANN()
    out = model(random_tensor)
    print(out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")