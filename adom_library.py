import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = './data'
models_root = './models'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def show_device():
    print(DEVICE)
    return

def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def classify_image_with_vgg_weights(image_path="tygrys.jpg", weights = models.VGG16_Weights.DEFAULT):
    vgg16 = models.vgg16(weights=weights)
    vgg16.eval()
    preprocess = weights.transforms()
    img = Image.open(image_path)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = vgg16(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    categories = weights.meta["categories"]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    plt.imshow(img)
    for i in range(5):
        print(categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2), "%")
    return

def classify_image_with_alexnet_weights(image_path="tygrys.jpg", weights = models.AlexNet_Weights.DEFAULT):
    alexnet = models.alexnet(weights=weights)
    alexnet.eval()
    preprocess = weights.transforms()
    img = Image.open(image_path)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = alexnet(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    categories = weights.meta["categories"]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    plt.imshow(img)
    for i in range(5):
        print(categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2), "%")
    return

def classify_image_with_googlenet_weights(image_path="tygrys.jpg", weights = models.GoogLeNet_Weights.DEFAULT):
    googlenet = models.googlenet(weights=weights)
    googlenet.eval()
    preprocess = weights.transforms()
    img = Image.open(image_path)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = googlenet(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    categories = weights.meta["categories"]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    plt.imshow(img)
    for i in range(5):
        print(categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2), "%")
    return

def classify_image_with_model(model, classes, image_path="tygrys.jpg"):
    img = Image.open(image_path)
    plt.imshow(img)
    img_tensor = transform(img)
    predict_top5_classes(model, img_tensor, classes)


def predict_top5_classes(model, image_tensor, classes):
    model = model.to(DEVICE)
    model.eval()
    image = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    top5_prob, top5_idx = torch.topk(probs, 5)

    for i in range(5):
        print(classes[top5_idx[0][i]], round(top5_prob[0][i].item()* 100, 2), "%")

def train_new_model(train_dataset, model_type="AlexNet", optimizer_name="Adam", epochs=10, batch_size=16, is_shuffle=True, with_train_output=True):
    num_classes = len(train_dataset.classes)
    model = get_model_without_weights(num_classes=num_classes,model_type=model_type)
    optimizer = get_optimizer_for_model(model=model, optimizer_name=optimizer_name)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=is_shuffle)
    return train_model(model, train_loader, optimizer, epochs, with_train_output)

def train_model(model, train_loader, optimizer, epochs, with_output):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for images, labels in train_loader:
            i=i+1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if(with_output):
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    return model

def get_model_without_weights(num_classes, model_type="VGG16"):
    if(model_type=="VGG16"):
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif(model_type=="AlexNet"):
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif(model_type=="GoogleNet"):
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(1024, num_classes)
    else:
        print("Type of model shuold be one of { VGG16, AlexNet, GoogleNet }")
    return model

def get_optimizer_for_model(model, optimizer_name="Adam"):
    if(optimizer_name=="Adam"):
        return optim.Adam(model.parameters(), lr=0.0001)
    elif(optimizer_name=="SGD"):
        return optim.SGD(model.parameters(), lr=0.01)
    else: 
        print("Name of optimizer should be one of { Adam, SGD }")

def get_CIFAR10_dataset():
    return datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)

def save_model(model, model_name, classes):
    torch.save({
    "model_state": model.state_dict(),
    "num_classes": len(classes),
    "class_names": classes
    }, models_root + "/" + model_name + ".pth")

def read_model(model_name, model_type):
    model_info = torch.load(models_root + "/" + model_name + ".pth")
    if(model_type=="VGG16"):
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, model_info["num_classes"])
    elif(model_type=="AlexNet"):
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, model_info["num_classes"])
    elif(model_type=="GoogleNet"):
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(1024, model_info["num_classes"])
    else:
        print("Type of model shuold be one of { VGG16, AlexNet, GoogleNet }")
    model.load_state_dict(model_info["model_state"])
    classes = model_info["class_names"]
    return model, classes