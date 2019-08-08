from utils import *
import torch
from torch.autograd.variable import Variable
import math
from IPython import display
import numpy as np


class Modeler:
    
    def __init__(self,epochs,batch_size,is_img_data=True):
        
        self.epochs=epochs
        self.batch_size=batch_size
        self.is_img_data=is_img_data
        

    def train(self,**kwargs_disc_gen):

        """

        :param kwargs_disc_gen: disc and gen config parameters
        :param is_img_data : signifies whether use image data
        :param disc_obj: disc being trained
        :param gen_obj: gen being trained

        """
        
        disc_obj=kwargs_disc_gen["dis_obj"]
        gen_obj=kwargs_disc_gen["gen_obj"]
        
        data_loader=None
        data=None
        if self.is_img_data:
            data = get_mnist_data()
        
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True) #get data in batches

        total_batches = len(data_loader)

        # --------------------------------------------- Training Function ----------------------------------------------

        test_images=get_noise(10,gen_obj.gen_in_dim) #test images for generator

        for epoch in range(self.epochs):
           
            for batch_num , (real_data, real_label) in enumerate(data_loader):
                #if batch_num==2:break
                batch_feats=real_data.shape[-1]
                batch_size=real_data.shape[0]
                #print(batch_size,batch_feats,real_data.shape)

                #--------------------------------- Training Discriminator -----------------------------

                if self.is_img_data:

                    real_data = itov(real_data,batch_feats)

                real_data=check_cuda_avail(Variable(real_data)) # real data for dis
                fake_data=get_noise(batch_size,gen_obj.gen_in_dim)
               
                gen_fake_out=gen_obj(fake_data) #forward pass fake data from generator

                kwargs_disc_gen["real_data"]=real_data #update dict real data with real data at present
                kwargs_disc_gen["fake_data"]=gen_fake_out #update dict fake data with fake data at present

                disc_loss, dis_real_out, disc_fake_out = disc_obj.train_disc(**kwargs_disc_gen) # train disc
                disc_obj.losses.append(disc_loss)
                #----------------------------------- Training Generator -----------------------------------------

                fake_data=get_noise(batch_size,gen_obj.gen_in_dim)
                kwargs_disc_gen["fake_data"] = fake_data  # update dict fake data with fake data at present
                kwargs_disc_gen["real_data"] = None  # clear memory

                gen_loss, gen_out=gen_obj.train_generator(**kwargs_disc_gen) #train generator
                gen_obj.losses.append(gen_loss)
                
                # ------------------------------------ Logging Models ------------------------------------------
                if batch_num%500 == 0: # save gen and dis models after every 500 batches

                    dis_path="./checkpoints_dis/epoch_{}_batch_{}".format(epoch,batch_num)+".pth"
                    gen_path = "./checkpoints_gen/epoch_{}_batch_{}".format(epoch, batch_num)+".pth"
                    save_model(disc_obj,dis_path) #saving disc
                    save_model(gen_obj,gen_path) #saving gen

                    #display test images for each epoch after every 100 batches

                    print("Epoch : {} Batch : {} Gen Loss : {} Dis Loss : {}".format(epoch, batch_num, gen_loss, disc_loss))
                    #display.clear_output(True)
                    test_images_out = gen_obj(test_images)
                    plot_image(test_images_out,batch_feats,epoch,batch_num)
             
        make_dir("./losses")
        np.save("./losses/gen_losses",np.array(gen_obj.losses))
        np.save("./losses/dis_losses",np.array(disc_obj.losses))
        