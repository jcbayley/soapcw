#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys
import pickle
import scipy.stats as st
from .models import CVAE
import gen_cw_frac as gen_cw
import gen_data
import copy
import corner
from tools import make_ppplot, loss_plot, track_plot
import shutil 


def train_batch(epoch, model, optimizer, device, batch, labels, pause = 0, train = True, ramp = 1.0, dist_type = "gaussian", freqs = None):
    model.train(train)
    if train:
        optimizer.zero_grad()
    length = float(batch.size(0))
    # calculate r2, q and r1 means and variances

    if freqs is not None:
        recon_loss, kl_loss = model.compute_loss(batch, labels, freqs, ramp)
    else:
        recon_loss, kl_loss = model.compute_loss(batch, labels, None, ramp)
    # calcualte total loss
    loss = recon_loss + ramp*kl_loss
    if train:
        loss.backward()
        # update the weights                                                                                                                              
        optimizer.step()

    return loss.item(), kl_loss.item(), -recon_loss.item()

def train(model, device, epochs, train_iterator, learning_rate, validation_iterator, ramp_start = -1,ramp_end =-1,save_dir = "./", dec_rate = 1.0, dec_start = 0, do_test=True, low_cut = 1e-12, length = 312, test_data = None, noise_std=0.01, chunk_load = False, dist_type = "gaussian", stmod = False, continue_train = True):

    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    train_losses = []
    kl_losses = []
    lik_losses = []
    val_losses = []
    val_kl_losses = []
    val_lik_losses = []
    train_times = []
    kl_start =ramp_start
    kl_end = ramp_end
    min_val_loss = np.inf
    prev_save_ep = 0

    if continue_train:
        with open(os.path.join(save_dir, "checkpoint_loss.txt"),"r") as f:
            old_epochs, train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses = np.loadtxt(f)
            old_epochs = list(old_epochs)
            train_losses = list(train_losses)
            kl_losses = list(kl_losses)
            lik_losses = list(lik_losses)
            val_losses = list(val_losses)
            val_kl_losses = list(val_kl_losses)
            val_lik_losses = list(val_lik_losses)
            
    start_train_time = time.time()
    for epoch in range(epochs):
        if continue_train:
            epoch = epoch + old_epochs[-1]

        model.train()

        if epoch > dec_start:
            adjust_learning_rate(learning_rate, optimizer, epoch - dec_start, factor=dec_rate, low_cut = low_cut)
            
        if stmod is False:
            ramp = 0.0
            if epoch>kl_start and epoch<=kl_end:
                ramp = (np.log(epoch)-np.log(kl_start))/(np.log(kl_end)-np.log(kl_start)) 
            elif epoch>kl_end:
                ramp = 1.0 
        else:
            ramp = 1.0
        

        # Training    

        temp_train_loss = 0
        temp_kl_loss = 0
        temp_lik_loss = 0
        it = 0
        total_time = 0

        #for local_batch, local_labels in train_iterator:
        for ind in range(len(train_iterator)):
            # Transfer to GPU            
            local_batch, local_labels, local_freqs = train_iterator[ind]
            local_batch, local_labels, local_freqs = torch.Tensor(local_batch).to(device), torch.Tensor(local_labels).to(device), torch.Tensor(local_freqs).to(device)
            start_time = time.time()
            train_loss,kl_loss,lik_loss = train_batch(epoch, model, optimizer, device, local_batch,local_labels, ramp=ramp, train=True, dist_type = dist_type, freqs = local_freqs)
            temp_train_loss += train_loss
            temp_kl_loss += kl_loss
            temp_lik_loss += lik_loss
            it += 1
            total_time += time.time() - start_time
        

        val_it = 0
        temp_val_loss = 0
        temp_val_kl_loss = 0
        temp_val_lik_loss = 0
        
        # validation
        #for val_batch, val_labels in validation_iterator:
        for ind in range(len(validation_iterator)):
            # Transfer to GPU            
            val_batch, val_labels, val_freqs = validation_iterator[ind]
            val_batch, val_labels, val_freqs = torch.Tensor(val_batch).to(device), torch.Tensor(val_labels).to(device), torch.Tensor(val_freqs).to(device)
            val_loss,val_kl_loss,val_lik_loss = train_batch(epoch, model, optimizer, device, val_batch, val_labels, ramp=ramp, train=False, dist_type = dist_type, freqs = val_freqs)
            temp_val_loss += val_loss
            temp_val_kl_loss += val_kl_loss
            temp_val_lik_loss += val_lik_loss
            val_it += 1

        temp_val_loss /= val_it
        temp_val_kl_loss /= val_it
        temp_val_lik_loss /= val_it

        temp_train_loss /= it
        temp_kl_loss /= it
        temp_lik_loss /= it
        batch_time = total_time/it
        post_train_time = time.time()
        
        val_losses.append(temp_val_loss)
        val_kl_losses.append(temp_val_kl_loss)
        val_lik_losses.append(temp_val_lik_loss)
        train_losses.append(temp_train_loss)
        kl_losses.append(temp_kl_loss)
        lik_losses.append(temp_lik_loss)
        train_times.append(post_train_time - start_train_time)

        diff_ep = epoch - prev_save_ep
        if temp_val_loss < min_val_loss and diff_ep > 400:
            torch.save(model, os.path.join(save_dir,"model.pt"))  # save the model
            min_val_loss = temp_val_loss#np.inf
            prev_save_ep = 0
        if epochs - epoch < 10:
            torch.save(model, os.path.join(save_dir,"model_epoch{}.pt".format(epoch)))  # save the model

        if epoch % 40 == 0:
            print("Train:      Epoch: {}, Training loss: {}, kl_loss: {}, l_loss:{}, Epoch time: {}, batch time: {}".format(epoch,temp_train_loss,temp_kl_loss,temp_lik_loss, total_time,batch_time))
            print("Validation: Epoch: {}, Training loss: {}, kl_loss: {}, l_loss:{}".format(epoch,temp_val_loss,temp_val_kl_loss,temp_val_lik_loss))
            loss_plot(save_dir, train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses)
            with open(os.path.join(save_dir, "checkpoint_loss.txt"),"w+") as f:
                if len(train_losses) > epoch+1:
                    epoch = len(train_losses) - 1
                savearr =  np.array([np.arange(epoch+1), train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses]).astype(float)
                np.savetxt(f,savearr)
            with open(os.path.join(save_dir, "checkpoint_times.txt"),"w+") as f:
                if len(train_times) > epoch+1:
                    t_epoch = len(train_times) - 1
                else:
                    t_epoch = epoch
                savearr =  np.array([np.arange(t_epoch+1), train_times]).astype(float)
                np.savetxt(f,savearr)
                
        
        if epoch % int(epochs/20) == 0:# and epoch > 3000:
            if do_test or epoch == epochs - 1:
                # test plots
                samples,tr,zr_sample,zq_sample = run_latent(model,validation_iterator,num_samples=1000,device=device)                                                   
                lat2_fig = latent_samp_fig(zr_sample,zq_sample,tr)
                lat2_fig.savefig('{}/zsample_epoch{}.png'.format(save_dir,epoch))
                del samples, tr, zr_sample
                plt.close(lat2_fig)
                if test_data is not None:
                    test_model(os.path.join(save_dir,"epochs_{}".format(epoch)), test_data, 20, copy.copy(model), length=length, noise_std = noise_std, num_samples = 1000, mid_train = True) 
                    torch.save(model, os.path.join(save_dir,"epochs_{}".format(epoch),"model.pt"))  # save the model
                    print("done_test")

        
        if epoch % 10 == 0 and chunk_load:
            train_iterator.load_new_chunk()

    return train_losses, kl_losses, lik_losses, val_losses, val_kl_losses, val_lik_losses
