import timm
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from get_loader import get_loader
from torchvision import transforms
from torch import Tensor
import math
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class EncoderMaxVit(nn.Module):
    def __init__(self,embedding_size,train_CNN=False,Model_Name='maxvit_rmlp_tiny_rw_256.sw_in1k',drop_p=0.5):
        super(EncoderMaxVit,self).__init__()
        self.embed_size = embedding_size
        self.train_CNN = train_CNN
        self.Model_Name = Model_Name
        self.p=drop_p
        
        self.model = timm.create_model(Model_Name,pretrained=True)
    
        self.model.head.fc = nn.Linear(in_features=self.model.head.fc.in_features , out_features= self.embed_size,bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)
        
        for name , param in self.model.named_parameters():
            if "head.fc.weight" in name or "head.fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def forward(self,image):
        output = self.model(image)
        output = self.dropout(self.relu(output))
        
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformersDecoder(nn.Module):
    def __init__(self,embeding_size,trg_vocab_size,num_heads,num_decoder_layers,dropout):
        super(TransformersDecoder,self).__init__()
        
        self.num_heads = num_heads
        self.embedding = nn.Embedding(trg_vocab_size,embeding_size)
        self.pos = PositionalEncoding(d_model = embeding_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embeding_size, nhead=num_heads)
        self.decoder= nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.device = "cuda"
        
        self.linear = nn.Linear(embeding_size , trg_vocab_size)
        self.drop = nn.Dropout(dropout)
        
        
    def make_mask(self,sz):
        mask = torch.zeros((sz,sz), dtype=torch.float32)
        for i in range(sz):
            for j in range(sz):
                if j > i: mask[i][j] = float('-inf')
        return mask
    
    def forward(self,features,caption):
        
        tgt_seq_length , N =caption.shape
        
        embed = self.drop(self.embedding(caption))
        embed = self.pos(embed)
        
        trg_mask = self.make_mask(tgt_seq_length).to(self.device)
        
        decoder = self.decoder(tgt = embed , memory = features.unsqueeze(0) , tgt_mask = trg_mask )
        
        output = self.linear(decoder)
        
        return output
        
        
        
        
class EncodertoDecoder(nn.Module):
    def __init__(self,embeding_size=512,trg_vocab_size=2993,num_heads=8,num_decoder_layers=6,dropout=0.2):
        super(EncodertoDecoder,self).__init__()
  
        self.encoder = EncoderMaxVit(embeding_size)
        
        self.decoder = TransformersDecoder(embeding_size=embeding_size,
                                           trg_vocab_size=trg_vocab_size,
                                           num_heads=num_heads,
                                           num_decoder_layers=num_decoder_layers,
                                           dropout=dropout)
        
    def forward(self , image , caption):
        
        features = self.encoder(image)
        output = self.decoder(features , caption)
        
        return output
        
        
        