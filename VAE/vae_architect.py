import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

args = {
    'gpu': 1,
    'epoch': 2,
    "batchsize": 32 ,
    "hiddensize": 64 ,
    "nthreads": 2,
    "epochs_mdn": 2,
    "nmix": 8,
    "logstep": 100,
    "dataset_key": "lfw"
}

def get_dirpaths(args):
    if args['dataset_key'] == 'lfw':
        listdir = "data/imglist/lfw"
        featslistdir = "data/featslist/lfw"
        out_dir = "data/output/lfw"
    
    else:
        raise NameError ("[ ERROR ] Incorrect key: %s" % ( args . dataset_key ) )
    return out_dir , listdir , featslistdir

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__(args)
        self.hidden_size = args['hiddensize']
        
        #encoder
        self.enc_conv1 = nn.Conv2d(2, 128, 5, stride=2, padding=2)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.enc_bn2 = nn.BatchNorm1d(256)
        self . enc_conv3 = nn. Conv2d (256 , 512 , 5 , stride =2 , padding =2)
        self . enc_bn3 = nn. BatchNorm2d (512)
        self . enc_conv4 = nn. Conv2d (512 , 1024 , 3 , stride =2 , padding =1)
        self . enc_bn4 = nn. BatchNorm2d (1024)
        self . enc_fc1 = nn. Linear (4*4*1024 , self . hidden_size *2)
        self . enc_dropout1 = nn. Dropout (p =0.7)

        # Conditional encoder block
        self.cond_enc_conv1 = nn.Conv2d(1, 128, 5, stride=2, padding=2)
        self.cond_enc_bn1 = nn.BatchNorm2d(128)
        self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.cond_enc_bn2 = nn.BatchNorm2d(256)
        self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.cond_enc_bn3 = nn.BatchNorm2d(512)
        self.cond_enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.cond_enc_bn4 = nn.BatchNorm2d(1024)

        #decoder block
        self.dec_upsamp1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.dec_conv1 = nn.Conv2d(1024+self.hidden_size, 512, 3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(512)
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv2 = nn.Conv2d(512*2, 256, 5, stride=1, padding=2)
        self.dec_bn2 = nn.BatchNorm2d(256)
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv3 = nn.Conv2d(256*2, 128, 5, stride=1, padding=2)
        self.dec_bn3 = nn.BatchNorm2d(128)
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv4 = nn.Conv2d(128*2, 64, 5, stride=1, padding=2)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv5 = nn.Conv2d(64, 2, 5, stride=1, padding=2)

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.enc_bn1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_bn2(x)
        x = F.relu(self.enc_conv3(x))
        x = self.enc_bn3(x)
        x = F.relu(self.enc_conv4(x))
        x = self.enc_bn4(x)
        x = x.view(-1, 4*4*1024)
        x = self.enc_dropout1(x)
        x = self.enc_fc1(x)
        mu = x[..., :self.hidden_size]
        logvar = x[..., self.hidden_size:]
        return mu, logvar
    
    def cond_encoder(self, x):
        x = F.relu(self.cond_enc_conv1(x))
        sc_feat32 = self.cond_enc_bn1(x)
        x = F.relu(self.cond_enc_conv2(sc_feat32))
        sc_feat16 = self.cond_enc_bn2(x)
        x = F.relu(self.cond_enc_conv3(sc_feat16))
        sc_feat8 = self.cond_enc_bn3(x)
        x = F.relu(self.cond_enc_conv4(sc_feat8))
        sc_feat4 = self.cond_enc_bn4(x)
        return sc_feat32, sc_feat16, sc_feat8, sc_feat4
    
    def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):
        x = z.view(-1, self.hidden_size, 1, 1)
        x = self.dec_upsamp1(x)
        x = torch.cat([x, sc_feat4], 1)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x)
        x = torch.cat([x, sc_feat8], 1)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)
        x = self.dec_upsamp3(x)
        x = torch.cat([x, sc_feat16], 1)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)
        x = self.dec_upsamp4(x)
        x = torch.cat([x, sc_feat32], 1)
        x = F.relu(self.dec_conv4(x))
        x = self.dec_bn4(x)
        x = self.dec_upsamp5(x)
        x = torch.tanh(self.dec_conv5(x))
        return x
    
    def forward(self , color , greylevel , z_in = None):
        sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
        mu,logvar = self.encoder(color)
        if self.training:
            stddev = torch.sqrt(torch.exp(logvar))
            #reparametrization
            eps = torch.randn_like(stddev)
            z = mu + eps * stddev
        else:
            z = z_in
            z = z.to(greylevel.device)
        color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
        return mu, logvar, color_out

def vae_loss(mu , logvar , pred , gt , lossweights , batchsize):
    kl_element = torch.add(torch.add(torch.add(mu.pow(2),logvar.exp()),-1),logvar.mul(-1))
    kl_loss = torch.mul(0.5,kl_element).mean()
    
    gt = gt.reshape(-1, 64*64*2)
    pred = pred.reshape(-1, 64*64*2)
    recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2),lossweights),1))
    recon_loss = torch.sum((recon_element).mul(1.0/batchsize))

    recon_element_l2 = torch . sqrt ( torch .sum ( torch .add(gt , pred .mul ( -1) ) . pow (2) ,1) )
    recon_loss_l2 = torch .sum ( recon_element_l2 ) .mul (1.0 / ( batchsize ) )

    return kl_loss, recon_loss, recon_loss_l2