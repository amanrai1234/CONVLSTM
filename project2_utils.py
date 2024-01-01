import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import matplotlib.pyplot as plt
import argparse
import random
import gzip

def save_img(img, args, name=''):
    npimg = img.numpy()
    npimg = Image.fromarray((npimg * 255).transpose(1,2,0).astype(np.uint8))
    npimg.save(os.path.join(args.ckpt_dir,'result%s.jpg'%name))

def show_video(x, y_hat, y):
    
    # predictions with input for illustration purposes
    preds = torch.cat([x.cpu(), y_hat.cpu()], dim=1)[0]

#     # entire input and ground truth
    y_plot = torch.cat([x.cpu(), y.cpu()], dim=1)[0]

#     # error (l2 norm) plot between pred and ground truth
#     difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
#     zeros = torch.zeros(difference.shape)
#     difference_plot = torch.cat([zeros.cpu(), difference.cpu()], dim=0)

    # concat all images
    final_image = torch.cat([preds, y_plot], dim=0)

    # make them into a single grid image file
    grid = torchvision.utils.make_grid(final_image, nrow=20, padding=0)

    return grid


def train(net, loader, criterion, optimizer, epoch, args):
    net.train()
    
    for i, batch in enumerate(loader):
        
        x, y = batch[0].to(args.device).float(), batch[1].to(args.device).float()
        
        net.zero_grad()
        y_hat = net.forward(x, args.n_steps_ahead).transpose(1,2)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        # print training progress every 50 global_step
        if i % 50 == 0:
            print('Epoch:[{}/{}], Step: [{}] '
                  'learning_rate: {}, '
                  'loss: {:.4f}'
                  .format(epoch, args.epochs, i, 
                          args.lr,
                          loss.item()))
       
def evaluate_epoch(net, loader, criterion, epoch, args):
    net.eval()
    
    Loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):

            x, y = batch[0].to(args.device), batch[1].to(args.device)
            y_hat = net.forward(x, args.n_steps_ahead).transpose(1,2) 
            loss = criterion(y_hat, y)
            
            Loss += loss.item()
            count += 1
        
        final_image = show_video(x, y_hat, y)
        save_img(final_image, args)
        
        avg_loss = Loss / count
        print('Average MSE loss on test dataset: {:.4f}'.format(avg_loss))
    return avg_loss
        
def evaluate(net, loader, criterion, args):
    net.eval()
    
    Loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):

            x, y = batch[0].to(args.device), batch[1].to(args.device)
            y_hat = net.forward(x, args.n_steps_ahead).transpose(1,2) 
            loss = criterion(y_hat, y)
            
            Loss += loss.item()
            count += 1

            final_image = show_video(x, y_hat, y)
            save_img(final_image, args, str(i))
        
        avg_loss = Loss / count
        print('Average MSE loss on test dataset: {:.4f}'.format(avg_loss))
    return avg_loss
    
def create_optimizer(net, learning_rate):
    return torch.optim.Adam(net.parameters(), lr=learning_rate)

def create_criterion():
    return torch.nn.MSELoss()

def checkpoint(net, epoch, cur_loss, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(net.state_dict(),
               '{}/net_{}'.format(args.ckpt_dir, suffix_latest))

    if cur_loss < args.best_loss:
        args.best_acc = cur_loss
        torch.save(net.state_dict(),
                       '{}/net_{}'.format(args.ckpt_dir, suffix_best))

def load_train_data(root):
    # Load train dataset for generating training data.
    with gzip.open(root, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
    return data

class MovingDigits(data.Dataset):
    def __init__(self, root, is_train, n_frames_input=10, n_frames_output=10, num_objects=[2]):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingDigits, self).__init__()

        self.dataset = None
        if is_train:
            self.digit = load_train_data(root)
        else:
            self.dataset = torch.load(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[0]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_digit(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.digit.shape[0] - 1)
            digit_image = self.digit[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_digit(num_digits)
            r = 1
            w = int(64 / r)
            images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        else:
            images = self.dataset[idx, ...].unsqueeze(1)

        input = images[:self.n_frames_input] 
        output = images[self.n_frames_input:length]

        if self.is_train:
            output = torch.from_numpy(output / 255.0).contiguous().float()
            input = torch.from_numpy(input / 255.0).contiguous().float()
        else:
            output = (output / 255.0).contiguous().float()
            input = (input / 255.0).contiguous().float()            

        return input, output

    def __len__(self):
        return self.length