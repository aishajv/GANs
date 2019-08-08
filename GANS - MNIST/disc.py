from torch import nn,optim
from utils import *


class Discriminator(nn.Module):
    """
    Discriminator class 
    """

    def __init__(self, **kwargs):

        """
        Default constructor

        Parameters:
        dis_in_dim (int) : dimension of date fed into discriminator
        dis_out_dim (int) : dimension of discriminator's output
        dis_hidden_layers (list) : list of neurons for each hidden layer
        leaky_relu (float) : lealy relu to be used for each hidden layer
        dropout (float) : dropout value to be used for each hidden layer
        """

        super(Discriminator, self).__init__()

        self.dis_in_dim = kwargs["dis_in_dim"]
        self.dis_out_dim = kwargs["dis_out_dim"]
        self.dis_hidden_layers = kwargs["dis_hidden_layers"]
        self.leaky_relu = kwargs["dis_leaky_relu"]
        self.dropout = kwargs["dis_dropout"]
        self.dis_layers = []  # stores hidden layers and output layer objects
        self.losses=[]
        self.create_dis()

    def create_dis(self):

        """
        Defines discriminator's architecture
        """
        # ------------------------------ buidling architecture -------------------------#
        input_dim = self.dis_in_dim

        for hidden_layer_neurons in self.dis_hidden_layers:  # building hidden layers

            hidden_layer = nn.Sequential(nn.Linear(input_dim, hidden_layer_neurons),
                                         nn.LeakyReLU(self.leaky_relu),
                                         nn.Dropout(self.dropout))
            
            self.dis_layers.append(hidden_layer)                      
            input_dim = hidden_layer_neurons

        # adding output layer
        output_layer = nn.Sequential(nn.Linear(input_dim, self.dis_out_dim),
                                     nn.Sigmoid())

        self.dis_layers.append(output_layer)
        self.dis_layers=nn.Sequential(*self.dis_layers)
        
    def forward(self, dataset):

        """

        Feed forward function for discriminator

        Parameters
        dataset (2D Array - N X D) : dataset for discriminator's feed forward

        Returns
        layer_output (2D Array - N X self.dis_out_dim) : output of discriminator

        """
        """layer_input = dataset
        layer_output = None
        for layer in self.dis_layers:
            layer_output = layer(layer_input)  # feed forward in this particular layer
            layer_input = layer_output

        return layer_output"""
        return self.dis_layers(dataset)

    def train_disc(self, **kwargs):

        """
        Function to train discriminator


        Parameters:

        real_data (2D array - N X D) : real data to be passed from discriminator
        fake_data (2D array - N X D) : fake data to be passed from discriminator
        optim_dis: optimizer of dis
        loss_dis: binary cross entropy loss object

        Returns
        total_loss : discriminator's total loss
        fake_out : forward pass of fake dataset
        real_out : forward pass of real dataset

        """

        real_data = kwargs["real_data"]
        fake_data = kwargs["fake_data"]
        optim_dis = kwargs["optim_dis"]
        loss_dis = kwargs["loss"]
        batch_size = real_data.shape[0]

        # clear any existing gradients of discriminator

        optim_dis.zero_grad()

        # ----------------- training discriminator on real data set ----------------------
        real_out = self.forward(real_data)  # feed forward using real data
        real_loss = loss_dis(real_out, get_real_data(batch_size, 1))
        real_loss.backward()

        # ----------------- training discriminator on fake data set ----------------------
        fake_out = self.forward(fake_data)  # feed forward using fake data
        fake_loss = loss_dis(fake_out, get_fake_data(batch_size, 1))
        fake_loss.backward()

        optim_dis.step() # updating weights based on real & fake data gradients

        total_loss = real_loss + fake_loss

        return total_loss, real_out, fake_out


