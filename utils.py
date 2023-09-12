import torch
import torchvision.transforms as transforms
from PIL import Image

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

def print_examples(model, device, dataset):
 
    transform = transforms.Compose([transforms.Resize((350,350)),
                               transforms.RandomCrop((256,256)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model.eval()
    test_img1 = transform(Image.open(r"M:\Python\Text Processing\test_examples\dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Two dogs of different breeds looking at each other on the road .")
    print(
        "Example 1 OUTPUT: "
        + " ".join(caption_generate(model,dataset,test_img1.to(device),device,max_length = 50))
    )
    print("\n")
    test_img2 = transform(
        Image.open(r"M:\Python\Text Processing\test_examples\child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: The girl in the red jacket is next to a picture of a funny face .")
    print(
        "Example 2 OUTPUT: "
        + " ".join(caption_generate(model,dataset,test_img2.to(device),device,max_length = 50))
    )
    test_img3 = transform(Image.open(r"M:\Python\Text Processing\test_examples\bike.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("\n")
    print("Example 3 CORRECT: Man on four wheeler in the air .")
    print(
        "Example 3 OUTPUT: "
        + " ".join(caption_generate(model,dataset,test_img3.to(device),device,max_length = 50))
    )
    print("\n")
    test_img4 = transform(
        Image.open(r"M:\Python\Text Processing\test_examples\hourse.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT:Two horses pulling a sled steered by a smiling blond woman .")
    print(
        "Example 4 OUTPUT: "
        + " ".join(caption_generate(model,dataset,test_img4.to(device),device,max_length = 50))
    )

    print("\n")
    test_img5 = transform(
        Image.open(r"M:\Python\Text Processing\test_examples\people.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: People gather on a bridge near a weeping willow tree .")
    print(
        "Example 5 OUTPUT: "
        + " ".join(caption_generate(model,dataset,test_img5.to(device),device,max_length = 50))
    )
    model.train()


def save_checkpoint(state, filename="M:\\Python\\Text Processing\\VIT Transformers\\Models\\my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("=> Checkpoint saved successfully")


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step