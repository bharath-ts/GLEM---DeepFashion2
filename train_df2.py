"""
A Global-local Embedding Module for Fashion Landmark Detection
ICCV 2019 Workshop 'Computer Vision for Fashion, Art, and Design'
"""
import json
import os
from arg import argument_parser
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import Network
from utils import cal_loss, Evaluator
import utils
from tqdm import tqdm, trange
import time

parser = argument_parser()
args = parser.parse_args()

def main():
    # random seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    from dataset_df2_loader import DeepFashionDataset as DataManager
    with open("./data/train/deepfashion2.json",'r') as infile:
            ds = json.load(infile)
            ds = ds['annotations'][0:5]

    print("dataset", len(ds), args.batchsize, args.epoch)
    
    print('dataset : %s' % (args.dataset[0]))
    if not args.evaluate:
        train_dm = DataManager(ds, root=args.root)
        train_dl = DataLoader(train_dm, batch_size=args.batchsize, shuffle=True)

        if os.path.exists('models') is False:
            os.makedirs('models')
   
    with open("./data/validation/deepfashion2_datafile_8.json",'r') as infile:
        test_data = json.load(infile)
   
    test_dm = DataManager(test_data['annotations'][0:5], root="/media/chintu/bharath_ext_hdd/Bharath/Segmentation/Landmark detection/GLE_FLD-master/data/validation/image/")    
    test_dl = DataLoader(test_dm, batch_size=args.batchsize, shuffle=False)



    # Load model
    print("Load the model...")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    net = torch.nn.DataParallel(Network(dataset=args.dataset, flag=args.glem)).to(device)
    if not args.weight_file == None:
        weights = torch.load(args.weight_file)
        if args.update_weight:
            weights = utils.load_weight(net, weights)
        net.load_state_dict(weights)

    # evaluate only
    if args.evaluate:
        print("Evaluation only")
        test(net, test_dl, 0)
        return

    # learning parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    print('Start training')
    for epoch in range(args.epoch):
        lr_scheduler.step()
        train(net, optimizer, train_dl, epoch)
        test(net, test_dl, epoch)


def train(net, optimizer, trainloader, epoch):

    train_step = len(trainloader)
    net.train()
    pbar = tqdm(trainloader)
    for i, sample in enumerate(pbar):
        iter_start_time = time.time()
        
        for key in sample:
            sample[key] = sample[key].cuda()
        
        output = net(sample)
        loss = cal_loss(sample, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t = time.time() - iter_start_time
        if (i + 1) % 10 == 0:
            
            tqdm.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{:.3f}'.format(epoch + 1, args.epoch, i + 1, train_step, loss.item(),t))
            
    save_file = 'model_%02d.pkl'
    print('Saving Model : ' + save_file % (epoch + 1))
    torch.save(net.state_dict(), './models/'+ save_file % (epoch + 1))


def test(net, test_loader, epoch):
    net.eval()
    test_step = len(test_loader)
    print('\nEvaluating...')
    with torch.no_grad():
        evaluator = Evaluator()
        pbar2 = tqdm(test_loader)
        for i, sample in enumerate(pbar2):
            iter_start_time = time.time()
            for key in sample:
                sample[key] = sample[key].cuda()

            output = net(sample)
            evaluator.add(output, sample)
            t = time.time() - iter_start_time
            if (i + 1) % 100 == 0:
                tqdm.write('Val Step [{}/{}],  Time:{:.3f}'.format(i + 1, test_step, t))
            
        results = evaluator.evaluate()
        print('Epoch {}/{}'.format(epoch + 1, args.epoch))
        print('lm_dist_all: {:.5f} '.format(results['lm_dist_all']))

        
if __name__ == '__main__':
    main()
