import timm
import numpy as np
from torch import nn
import torch
import torch.optim as optim
import os
import random

from get_loader import get_loader
from torchvision import transforms
from torch import Tensor
import math
from PIL import Image
from Model import EncodertoDecoder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, print_examples
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.Resize((350,350)),
                                transforms.RandomCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images_path , caption_path = r"M:\Python\Text Processing\Dataset\Images" , r"M:\Python\Text Processing\Dataset\captions.txt"

data_loader , dataset = get_loader(images_path,caption_path ,transform,batch_size = 32,num_workers=2)

def caption_generate(model,dataset,image,device,max_length = 50):
    outputs=[dataset.vocab.stoi["<SOS>"]]
    for i in range(max_length):
        trg_tensor =torch.LongTensor(outputs).unsqueeze(1).to(device)
        image = image.to(device)
        
        with torch.no_grad():
            output = model(image,trg_tensor)
            
        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)
        
        if best_guess == dataset.vocab.stoi["<EOS>"]:
            break
    caption = [dataset.vocab.itos[idx] for idx in outputs]
    
    return caption[1:]

torch.backends.cudnn.benchmark = True

device=torch.device("cuda")
n_token = len(dataset.vocab)
d_model = 256
embedding_size=256
n_head = 8
learning_rate = 3e-4
num_epochs = 100
step = 0
save_model=True
vocab_size = len(dataset.vocab)
print("This is Vocab size",len(dataset.vocab))
num_layers = 1
learning_rate = 3e-4
num_epochs = 100

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
loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])

checkpoint=torch.load(r"M:\Python\Text Processing\VIT\my_checkpoint.pth.tar")
mymodel=model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])
# step = checkpoint["step"]
device=torch.device("cuda")



def print_examples(model, device, dataset,image):
    transform = transforms.Compose([transforms.Resize((350,350)),
    transforms.RandomCrop((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    model.eval()
    image_path=image
    print(image)
    test_img1 = transform(Image.open(image).convert("RGB")).unsqueeze(0)
    output=caption_generate(model,dataset,test_img1.to(device),device,max_length = 50)
    print("Example 1 OUTPUT: "+ " ".join(output))
    plt.imshow(Image.open(image).convert("RGB"))
    plt.show()




path = r"M:\Python\Text Processing\Dataset\Images"
lists = os.listdir(path)
rand = random.choice(lists)
image = os.path.join(path,rand)

print_examples(model,device,dataset,image)
