import timm
import numpy as np
import torch
import torch.optim as optim
from get_loader import get_loader
from torchvision import transforms
from torch import Tensor
import math
from Model import EncodertoDecoder
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples

def train():

    transform = transforms.Compose([transforms.Resize((350,350)),
                                transforms.RandomCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    images_path , caption_path = r"M:\Python\Text Processing\Dataset\Images" , r"M:\Python\Text Processing\Dataset\captions.txt"
    BATCH_SIZE = 32
    data_loader , dataset = get_loader(images_path,caption_path ,transform,batch_size = BATCH_SIZE,num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5
    learning_rate = 3e-4
    trg_vocab_size = len(dataset.vocab)
    embedding_size = 512
    num_heads = 8
    num_decoder_layers = 4
    dropout = 0.10
    pad_idx=dataset.vocab.stoi["<PAD>"]
    load_model = False
    save_model = True
    writer =SummaryWriter("runs/loss_plot")
    step = 0

    model =EncodertoDecoder(embeding_size=embedding_size,
                            trg_vocab_size=trg_vocab_size,
                            num_heads=num_heads,
                            num_decoder_layers=num_decoder_layers,
                            dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    pad_idx = pad_idx

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


    for epoch in range(num_epochs):
        
        print(f"[Epoch {epoch} / {num_epochs}]")
        
        if save_model:
            checkpoint = {"state_dict" : model.state_dict(),"optimizer"  : optimizer.state_dict()}
            save_checkpoint(checkpoint)
            
        model.eval()
        print_examples(model, device, dataset)
        model.train()
        total_loss = 0.0
        for idx, (images, captions) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            images = images.to(device)
            captions = captions.to(device)
            
            output = model(images, captions[:-1])
            output = output.reshape(-1, output.shape[2])
            target = captions[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output,target)
            lossofepoch = loss.item()
            total_loss += lossofepoch
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            
            optimizer.step()
            writer.add_scalar("Training Loss",loss,global_step=step)
            step+=1
            
        print("Loss of the epoch is", total_loss / len(data_loader))



if __name__ == "__main__":
    train()
            
        
        