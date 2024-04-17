import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from utils.data import MyDataset
import sys 
import random
import argparse
from models import model
import torch.nn as nn
import torch
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Inference Table Understanding")
    parser.add_argument('--lr', type=float, default=1e-5,help='Learning rate')
    parser.add_argument('--e', type=int, default=3, help='The number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', type=int, default='bert-base-uncased', help='Model name')
    parser.add_argument('--max_seq', type=int, default='64', help='Sequence length')
    parser.add_argument('--temp', type=float, default='0.2', help='Temperature for cse')
    parser.add_argument('--data_path', type=str, default='/data/dataset.json', help='Dataset path')

    args = parser.parse_args()
    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model(model_name, args.temp)

    data={}
    with open(args.data_path, 'w') as json_file:
        json.dump(data, json_file)
        
    mydataset = MyDataset(data['x'],data['y'], args.max_seq, tokenizer)
    
    # Create a PyTorch Dataset
    dataset = torch.utils.data.TensorDataset(mydataset.x[:,:]['input_ids'],mydataset.x[:,:]['attention_masks'])

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = nn.Adam(model.parameters(), lr=0.01)

    for epoch in args.epochs:
        epoch_loss=0.0
        
        for idx, batch in enumerate(dataloader):
            x_input_ids, x_attention_mask = batch
            x_input_ids, x_attention_mask = x_input_ids.to(device), x_attention_mask.to(device)

            z1, z2 = model(x_input_ids, x_attention_mask)

            cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cos_sim, labels)
            
            optimizer.zero_grad()  
            loss.backward() 
            optimizer.step() 
            
            if (idx + 1) % 300 == 0:
                average_loss = epoch_loss / 300
                print("Epoch [{}/{}], Iteration [{}/{}], Average Loss: {:.4f}".format(epoch + 1, args.epochs, idx+ 1, len(dataloader), average_loss))
                epoch_loss = 0.0