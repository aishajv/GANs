from torch import nn,optim
from utils import *
import os


class Generator(nn.Module):
    """
    Generator class 
    """

    def __init__(self, **kwargs):

        """
        
        Default constructor
        
        Parameters:
        gen_in_dim (int) : dimension of noise fed into generator
        gen_out_dim (int) : dimension of generator's output
        gen_hidden_layers (list) : list of neurons for each hidden layer
        leaky_relu (float) : lealy relu to be used for each layers
        """

        super(Generator, self).__init__()

        self.gen_in_dim = kwargs["gen_in_dim"]
        self.gen_out_dim = kwargs["gen_out_dim"]
        self.gen_hidden_layers = kwargs["gen_hidden_layers"]
        self.leaky_relu = kwargs["gen_leaky_relu"]
        self.gen_layers = []  # stores hidden layers and output layer objects
        self.losses=[]
        self.create_gen()
        

    def create_gen(self):

        """
        Defines generator's architecture
        """

        # ------------------------------ building architecture -------------------------#
        input_dim = self.gen_in_dim

        for hidden_layer_neurons in self.gen_hidden_layers:  # building hidden layers

            hidden_layer = nn.Sequential(nn.Linear(input_dim, hidden_layer_neurons),
                                         nn.LeakyReLU(self.leaky_relu))

            self.gen_layers.append(hidden_layer)
            input_dim = hidden_layer_neurons

        # adding output layer
        output_layer = nn.Sequential(nn.Linear(input_dim, self.gen_out_dim),
                                     nn.Tanh())

        self.gen_layers.append(output_layer)
        self.gen_layers=nn.Sequential(*self.gen_layers)

    def forward(self, dataset):

        """
        
        Feed forward function for generator 
        
        Parameters
        dataset (2D Array - N X D) : dataset for generator's feed forward 
        
        Returns
        layer_output (2D Array - N X self.gen_out_dim) : output of generator
        
        """
        """layer_input = dataset
        layer_output = None
        for layer in self.gen_layers:
            layer_output = layer(layer_input)  # feed forward in this particular layer
            layer_input = layer_output

        return layer_output"""
        
        return self.gen_layers(dataset)

    def train_generator(self,**kwargs):

        """
        Function that train generator

        Parameters:
        optim_gen: generator's optimizer
        fake_data: fake_data (2D array - N X D) : fake data to be passed from generator
        dis_obj: discriminator being used in GAN

        Returns:
        fake_loss : generator's loss on output
        fake_out : forward pass of fake dataset
        """

        optim_gen=kwargs["optim_gen"]
        dis_obj=kwargs["dis_obj"]
        fake_data=kwargs["fake_data"]
        loss_gen=kwargs["loss"]
        batch_size=fake_data.shape[1]
        optim_gen.zero_grad() # clear any existing gradients

        gen_out=self.forward(fake_data)  #forward pass from generator
        dis_out=dis_obj(gen_out) #pass generator's output from discriminator
        dis_loss=loss_gen(dis_out,get_real_data(batch_size, 1)) #get loss value
        dis_loss.backward() # back propagate loss into discriminator and generator

        optim_gen.step() # update weights

        return dis_loss, gen_out

    def load_gen(self,model_dir): #returns latest checkpoint model 
        
        ckpts = os.listdir(model_dir)
        
        models=[]
        for model_name in ckpts: 
            if model_name.endswith(".pth"):
                models.append(model_dir+"/"+model_name)
       
        
        
        latest_model=max(models, key=os.path.getctime)
        
        print("Loading Latest Checkpoint File Model : {}".format(latest_model))
        
        ckpt=torch.load(latest_model)
        
        
        #return gen_obj.load_state_dict(ckpt)
        self.load_state_dict(ckpt)
    