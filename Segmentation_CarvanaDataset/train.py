import torch
import torch.nn as nn
import torch.utils.data as dt
from carvana_dataset import CarvanaDataset
from model import SegmenterModel
from torch.autograd import Variable
import torch.optim as optim
import os
import tqdm
import numpy as np

def two_hoot_coder(t):
    ones = torch.ones(t.shape)
    temp = ones - t
    return torch.cat((t, temp), dim = 1)

def trainig(net, n_epoch):
    useCuda =True
    train = './data/train/'
    train_masks = './data/train_masks/'
    test = './data/test/'
    test_masks = './data/test_masks'
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5, amsgrad=True)

    if useCuda == True:
        net = net.cuda()
        criterion= criterion.cuda()

    ds      = CarvanaDataset(train, train_masks)
    ds_test = CarvanaDataset(test, test_masks)

    dl      = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=16)
    dl_test = dt.DataLoader(ds_test, shuffle=False, num_workers=4, batch_size=16)

    train_loss = []
    test_loss = []
    for epoch in tqdm.tqdm_notebook(range(0, n_epoch)):
        epoch_train_loss = 0
        net.train(True)
        if epoch == 70:
            optimizer = optim.Adam(net.parameters(), lr = 0.0005)
        elif epoch == 90:
            optimizer = optim.Adam(net.parameters(), lr = 0.0001)
            
        for iter_, (i, t) in enumerate(dl):
            i = Variable(i)
            t = two_hoot_coder(t)
            t = Variable(t)
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = net(i)
            loss = criterion(o, t)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.data[0]

        epoch_train_loss = epoch_train_loss / float(len(ds))
        train_loss.append(epoch_train_loss)

        epoch_test_loss = 0
        net.train(False)
        for iter_, (i, t) in enumerate(dl_test):
            i = Variable(i, volatile = True)
            t = two_hoot_coder(t)
            t = Variable(t, volatile = True)
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = net(i)
            loss = criterion(o, t)
            epoch_test_loss += loss.data[0]
            
        epoch_test_loss = epoch_test_loss / float(len(ds_test))
        test_loss.append(epoch_test_loss)
        
        print("Current epoch: {}    Train loss = {:.5f}    Test Loss = {:.5f}".format(epoch, epoch_train_loss, epoch_test_loss))
        #print("-----------------------------------------------------------------")
    return train_loss, test_loss