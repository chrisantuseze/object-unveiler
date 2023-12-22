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

    netG = Generator(n_channel, z_size, img_size, class_num).to(device)
    netD = Discriminator(n_channel, img_size, class_num, batch_size).to(device)


    optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate,betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate,betas=(0.5, 0.999))

    netD.train()
    netG.train()

    criterion = nn.BCELoss()

    real_label = torch.ones([batch_size, 1], dtype=torch.float).to(device)
    fake_label = torch.zeros([batch_size, 1], dtype=torch.float).to(device)


    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}/{epochs}")
        for i, (input_sequence, label) in enumerate(data_loader):
            
            fixed_noise = torch.randn(batch_size, z_size, 1, 1, device=device)

            input_sequence = input_sequence.to(device)
            label_embed = label.to(device)
            
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
            new_label = torch.LongTensor(batch_size, class_num).random_(0, class_num).to(device)
            new_embed = new_label[:, 0].view(-1)
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
            
            if i % 100 == 0:
                print("D_loss:%f\tG_loss:%f" % (d_train_loss, g_train_loss))

        # Set generator eval
        netG.eval()

        # Building z
        # z = torch.randn((class_num, z_size), requires_grad=True).to(device)

        z = torch.randn(batch_size, z_size, 1, 1, device=device)

        # Labels 0 ~ 9
        # labels = Variable(torch.LongTensor(np.arange(class_num))).to(device)
        labels = torch.LongTensor(batch_size, class_num).random_(0, class_num).to(device)
        labels = new_label[:, 0].view(-1)

        # Generating images
        sample_images = netG(z, labels).unsqueeze(1).data.cpu()
        print("sample_images.shape", sample_images.shape)

        for i, image in enumerate(sample_images.squeeze(1)):
            image_path = os.path.join(output_folder, f'image_{i + 1}.png')
            save_image(torch.tensor(image), image_path)