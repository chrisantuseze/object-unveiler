from cgan.dataset import FashionMNIST
from cgan.model2 import Discriminator, Generator
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os
from PIL import Image

def generator_train_step(batch_size, z_size, class_num, device, discriminator, generator, g_optimizer, criterion):

    # Init gradient
    g_optimizer.zero_grad()

    # Building z
    z = torch.randn((batch_size, z_size), requires_grad=True).to(device)

    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)

    # Generating fake images
    fake_images = generator(z, fake_labels)

    # Disciminating fake images
    validity = discriminator(fake_images, fake_labels)

    # Calculating discrimination loss (fake images)
    g_loss = criterion(validity, torch.ones((batch_size, 1), requires_grad=True).detach().to(device))

    # Backword propagation
    g_loss.backward()

    #  Optimizing generator
    g_optimizer.step()

    return g_loss.data

def discriminator_train_step(batch_size, z_size, class_num, device, discriminator, generator, d_optimizer, criterion, real_images, labels):

    # Init gradient
    d_optimizer.zero_grad()

    # Disciminating real images

    noise = torch.randn(real_images.shape, requires_grad=True).to(device) * 0.1 # small variance
    
    img_noisy = real_images + noise

    real_validity = discriminator(img_noisy, labels)

    # Calculating discrimination loss (real images)
    real_loss = criterion(real_validity, torch.ones((batch_size, 1), requires_grad=True).detach().to(device))

    # Building z
    z = torch.randn((batch_size, z_size), requires_grad=True).to(device)

    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)

    # Generating fake images
    fake_images = generator(z, fake_labels)

    # Disciminating fake images
    fake_validity = discriminator(fake_images, fake_labels)

    # Calculating discrimination loss (fake images)
    fake_loss = criterion(fake_validity, torch.zeros((batch_size, 1), requires_grad=True).detach().to(device))

    # Sum two losses
    d_loss = real_loss + fake_loss

    # Backword propagation
    d_loss.backward()

    # Optimizing discriminator
    d_optimizer.step()

    return d_loss.data

def discriminate(model, img, y):
    noise = torch.randn(img.shape, requires_grad=True) * 0.1 # small variance
    
    img_noisy = img + noise
    return model(img_noisy, y)

def train():
    # Data
    train_data_path = 'save/fashion-mnist_train.csv' # Path of data
    print('Train data path:', train_data_path)

    img_size = 28 # Image size
    batch_size = 32  # Batch size

    # Model
    z_size = 100

    # Training
    epochs = 30  # Train epochs
    learning_rate = 1e-4

    class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_num = len(class_list)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    dataset = FashionMNIST(train_data_path, img_size, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define generator
    generator = Generator(z_size, img_size, class_num).to(device)
    # Define discriminator
    discriminator = Discriminator(img_size, class_num, batch_size).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Create a folder to save the images if it doesn't exist
    output_folder = 'save/output_images'
    os.makedirs(output_folder, exist_ok=True)

    for epoch in range(epochs):

        print('Starting epoch {}...'.format(epoch+1))

        for i, (images, labels) in enumerate(data_loader):

            # Train data
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Set generator train
            generator.train()

            # Train discriminator
            d_loss = discriminator_train_step(len(real_images), z_size, class_num, device, discriminator,
                                            generator, d_optimizer, criterion, real_images, labels)

            # Train generator
            g_loss = generator_train_step(batch_size, z_size, class_num, device, discriminator, generator, g_optimizer, criterion)

        # Set generator eval
        generator.eval()

        print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))

        # Building z
        z = torch.randn((class_num, z_size), requires_grad=True).to(device)

        # Labels 0 ~ 9
        labels = Variable(torch.LongTensor(np.arange(class_num))).to(device)

        # Generating images
        sample_images = generator(z, labels, bs=labels.shape[0]).unsqueeze(1).data.cpu()
        print(sample_images.shape)

        # Show images
        # grid = make_grid(sample_images.squeeze(1), nrow=3, normalize=True).permute(1,2,0).numpy()
        # plt.imshow(grid)
        # plt.show()

        # Save each image separately in the folder
        # for i, image in enumerate(grid):
        for i, image in enumerate(sample_images.squeeze(1)):
            image_path = os.path.join(output_folder, f'image_{i + 1}.png')
            save_image(torch.tensor(image), image_path)

            # Convert the image to a PIL Image before saving
            # pil_image = Image.fromarray((image * 255).astype(np.uint8))
            # pil_image.save(image_path)