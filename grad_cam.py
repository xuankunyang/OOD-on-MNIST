import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from NN_class import LeNet
from torch import cuda
from keras.preprocessing import image

def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    loaded_params = torch.load('save_model\\best_model_on_12_1.pth')
    model = LeNet().to(device)
    model.load_state_dict(loaded_params)
    target_layers = [model.conv2]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[1.0,1.0,1.0])
    ])

    # Prepare image
    img_path = "test_1.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = image.load_img(img_path, target_size=(28, 28))
    """data = torch.load('data\\data_final\\data_all_12.pt')
    id = 85552
    img = data[id][0]
    print(np.argmax(data[id][1]))"""
    img = np.array(img, dtype=np.float32)
    img_tensor = data_transform(img)
    """print(torch.max(img_tensor))"""
    img_tensor = img_tensor.permute(0,2,1)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    print(input_tensor.shape)
    output = model(input_tensor.to('cuda'))
    output = torch.squeeze(output)
    y_pred = torch.argmax(output)
    print(y_pred)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
