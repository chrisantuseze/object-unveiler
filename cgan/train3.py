from cgan.dataset import FashionMNIST
from cgan.model3 import Discriminator, Generator
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import os
from PIL import Image

def train():
    # Data
    batch_size = 32  # Batch size

    # Model
    z_size = 100

    # Training
    # epochs = 500  # Train epochs
    learning_rate = 2e-4

    # batchsize = 200
    epochs = 500
    class_num = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_type = "mnist"
    train_data_path = 'save/' # Path of data

    # Create a folder to save the images if it doesn't exist
    output_folder = 'save/output_images'
    os.makedirs(output_folder, exist_ok=True)

    n_channel = 1
    
    if data_type == "fashion_mnist":
        train_data_path = 'save/fashion-mnist_train.csv' # Path of data

        img_size = 28
        class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, ), std=(0.5, ))
        ])
        dataset = FashionMNIST(train_data_path, img_size, transform=transform)

    elif data_type == "mnist":
        img_size = 64
        dataset = datasets.MNIST(root=train_data_path, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ]))
        
    elif data_type == 'cifar10':
        img_size = 32
        dataset = datasets.CIFAR10(root=train_data_path, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        n_channel = 3

    print('Train data path:', train_data_path)

    # train_data = Data(file_name="cifar-10-batches-py/")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    netG = Generator(n_channel, z_size, img_size, class_num, batch_size).to(device)
    netD = Discriminator(n_channel, img_size, class_num, batch_size).to(device)


    optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate,betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate,betas=(0.5, 0.999))

    netD.train()
    netG.train()

    criterion = nn.BCELoss()

    real_label = torch.ones([batch_size, 1], dtype=torch.float).to(device)
    fake_label = torch.zeros([batch_size, 1], dtype=torch.float).to(device)

    g_early_stopper = EarlyStopper(patience=3, min_delta=10)
    d_early_stopper = EarlyStopper(patience=3, min_delta=10)

    for epoch in range(epochs):
        g_loss = 0.0
        d_loss = 0.0

        for i, (input_sequence, label) in enumerate(data_loader):
            
            fixed_noise = torch.randn(batch_size, z_size, 1, 1, device=device)

            input_sequence = input_sequence.to(device)
            label_embed = label.to(device)
            # print("label_embed.shape", label_embed.shape, "label_embed", label_embed)
            
            '''
                Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            '''

            d_output_real = netD(input_sequence, label_embed)
            d_output_real = d_output_real.view(batch_size, -1)
            d_real_loss = criterion(d_output_real, real_label)

            g_output = netG(fixed_noise, label_embed)

            d_output_fake = netD(g_output, label_embed)
            d_output_fake = d_output_fake.view(batch_size, -1)
            d_fake_loss = criterion(d_output_fake, fake_label)

            # Back propagation
            d_train_loss = (d_real_loss + d_fake_loss) / 2

            netD.zero_grad()
            d_train_loss.backward()
            optimizerD.step()

            '''
                Update G network: maximize log(D(G(z)))
            '''
            new_label = torch.LongTensor(batch_size, 1).random_(0, class_num).to(device)
            new_embed = new_label.view(-1)
            # print("new_embed.shape", new_embed.shape)

            g_output = netG(fixed_noise, new_embed)

            d_output_fake = netD(g_output, new_embed)
            d_output_fake = d_output_fake.view(batch_size, -1)
            g_train_loss = criterion(d_output_fake, real_label)

            # Back propagation
            netD.zero_grad()
            netG.zero_grad()
            g_train_loss.backward()
            optimizerG.step()
            
            if i % 500 == 0:
                print("G_Loss: %f\tD_Loss: %f" % (g_train_loss, d_train_loss))

            _g_loss = torch.sum(g_train_loss)
            _d_loss = torch.sum(d_train_loss)

            g_loss += _g_loss.detach().cpu().numpy()
            d_loss += _d_loss.detach().cpu().numpy()

        g_loss = g_loss/i
        d_loss = d_loss/i
        print(f"Epoch: {epoch}/{epochs}", "\t\tG_Loss: %f\tD_Loss: %f\n" % (g_loss, d_loss))

        # Set generator eval
        netG.eval()
        z = torch.randn(class_num, z_size, 1, 1, device=device)

        # Labels 0 ~ 9
        labels = torch.LongTensor(class_num, 1).random_(0, class_num).view(-1).to(device)
        # print("labels.shape", labels.shape, "labels", labels)

        # Generating images
        sample_images = netG(z, labels).unsqueeze(1).data.cpu()
        for i, image in enumerate(sample_images.squeeze(1)):
            image_path = os.path.join(output_folder, f'image_{labels[i]}.png')
            save_image(image, image_path)

        if g_early_stopper.early_stop(g_loss) or d_early_stopper.early_stop(d_loss):      
            break

def generate_dataset(generator, data_dir, n_classes, z_size):
    # Create a folder to save the images if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator.eval()
    for i in range(200):
        z = torch.randn(n_classes, z_size, 1, 1, device=device)
        labels = torch.LongTensor(n_classes, 1).random_(0, n_classes).view(-1).to(device)

        sample_images = generator(z, labels).unsqueeze(1).data.cpu()
        for i, image in enumerate(sample_images.squeeze(1)):
            image_path = os.path.join(data_dir, f'{i}/image_{labels[i]}.png')
            save_image(image, image_path)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0

        elif abs(loss - self.min_loss) < 1e-4: #(self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
