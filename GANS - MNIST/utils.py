import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from torch.autograd.variable import Variable
import numpy as np


def plot_losses(file_path="./losses"):
    
    files=os.listdir("./losses")
    width = 5
    height = 8
    
    fig = plt.figure(figsize=(height, width))
    for file_ in files:
        loss_array=np.load(file_path+"/"+file_,allow_pickle=True)
        
        label="Discriminator"
        if "gen" in file_ : label="Generator"
        
        
        plt.plot(loss_array,label=label)
    
    plt.xlabel="Epoch"
    plt.ylabel="Loss"
    plt.legend(loc='upper left')
    plt.show()
        
def make_dir(dir_path):
   
    dir_path=dir_path.split("/")[1]
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def plot_image(vector_images, reshape_size, epoch_no=None, batch_no=None, save_img=True, test=False):
   
   
    if torch.cuda.is_available():
        vector_images = vector_images.detach().reshape(-1,reshape_size, reshape_size).cpu()

    width = 12
    height = 12
    fig = plt.figure(figsize=(height, width))
    columns = int(vector_images.shape[0] / 2)
    rows = columns
    for i in range(1, vector_images.shape[0] + 1):
        img = vector_images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img,cmap="binary")

        path="./gans_output_{}/"
        
        if test: path=path.strip("/").format("test")+"/test_image"
        else : path=path.format("train") + "epoch_num_" + str(epoch_no) + "_batch_num_" +str(batch_no)
        
        make_dir(path)
        #print(path)
        plt.savefig(path)

    plt.show()


def save_model(model, model_path):
    """
    saves trained model in corresponding model path

    :param model: model to be saved
    :param model_path: path where model is saved
    """

    make_dir(model_path)
   
    torch.save(model.state_dict(), model_path)


def vtoi(vect_array, reshape_dims):
    """
    vector to image fucntion that reshapes given 2D array into an ND array of images

    :param vect_array: 2D vector array fo images
    :param reshape_dims: reshape dimension to reshape given vect_array into
    :return: reshaped ND array of images
    """

    return vect_array.view(vect_array.size(0), 1, reshape_dims, reshape_dims)


def itov(img_array, reshape_dim):
    """
    image to vector function that reshapes given ND image array into a 2D array

    :param reshape_dim: single d dimension to reshape every given image into
    :param img_array: given image array to be reshaped
    :return img_reshaped: reshaped array of images
    """
    
    return img_array.view(img_array.size(0), reshape_dim*reshape_dim)


def text_data():
    pass


def get_mnist_data(save_data_path="./dataset"):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
         ])

    return datasets.MNIST(root=save_data_path, train=True, download=True, transform=compose)


def check_cuda_avail(tensor_arg):
    """
    Simple check if cuda is available, and deploy tensor_arg on cuda

    Parameters:
    tensor_arg: tensor to be deployed on cuda

    Returns:
    tensor_arg: deployed/not deployed tensor

    """

    if torch.cuda.is_available():
        return tensor_arg.cuda()

    return tensor_arg


def get_noise(num_examples, noise__out_dim):
    """
    Generates random noise from normal distribution for feeding into generator

    Parameters:
    num_examples (int) : number of noise examples for which to generate noise
    noise_out_dim (int) : output dimension for noise examples

    Returns
    noise_out (2D array num_examples X noise_out_dim) : generated noise

    """

    noise_out = Variable(torch.randn(num_examples, noise__out_dim))

    return check_cuda_avail(noise_out)


def get_real_data(num_examples, out_dim):
    """
        Generates 2D tensor of ones

        Parameters:
        num_examples (int) : number of examples for which to generate ones
        out_dim (int) : output dimension of ones tensor

        Returns
        real_data (2D array num_examples X out_dim) : generated tensor of ones

    """

    real_data = Variable(torch.ones(num_examples, out_dim))

    return check_cuda_avail(real_data)


def get_fake_data(num_examples, out_dim):
    """
        Generates 2D tensor of zeros

        Parameters:
        num_examples (int) : number of examples for which to generate zeros

        Returns
        real_data (2D array num_examples X out_dim) : generated tensor of zeros

        """

    fake_data = Variable(torch.zeros(num_examples, out_dim))

    return check_cuda_avail(fake_data)
