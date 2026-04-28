import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset
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


def classify_image_with_vgg_weights(image_path="external_images/tygrys.jpg", weights=models.VGG16_Weights.DEFAULT):
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
        print(categories[top5_catid[i]], round(
            top5_prob[i].item() * 100, 2), "%")
    return


def classify_image_with_alexnet_weights(image_path="external_images/tygrys.jpg", weights=models.AlexNet_Weights.DEFAULT):
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
        print(categories[top5_catid[i]], round(
            top5_prob[i].item() * 100, 2), "%")
    return


def classify_image_with_googlenet_weights(image_path="external_images/tygrys.jpg", weights=models.GoogLeNet_Weights.DEFAULT):
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
        print(categories[top5_catid[i]], round(
            top5_prob[i].item() * 100, 2), "%")
    return


def classify_image_with_model(model, classes, image_path="external_images/tygrys.jpg"):
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
        print(classes[top5_idx[0][i]], round(
            top5_prob[0][i].item() * 100, 2), "%")


def train_new_model(train_dataset, model_type="AlexNet", optimizer_name="Adam", epochs=10, batch_size=16, is_shuffle=True, with_train_output=True, dropout_value=None):
    num_classes = len(train_dataset.classes)
    model = get_model_without_weights(
        num_classes=num_classes, model_type=model_type)

    model = set_dropout(model, model_type, dropout_value)

    optimizer = get_optimizer_for_model(
        model=model, optimizer_name=optimizer_name)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=is_shuffle)
    return train_model(model, train_loader, optimizer, epochs, with_train_output)


def train_model(model, train_loader, optimizer, epochs, with_output):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        i = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            i = i+1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if (with_output):
            print(
                f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")

    return model


def get_model_without_weights(num_classes, model_type="VGG16"):
    if (model_type == "VGG16"):
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif (model_type == "AlexNet"):
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif (model_type == "GoogleNet"):
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(1024, num_classes)
    else:
        print("Type of model shuold be one of { VGG16, AlexNet, GoogleNet }")
    return model


def get_optimizer_for_model(model, optimizer_name="Adam", lr=0.0001, momentum=0.9):
    if (optimizer_name == "Adam"):
        return optim.Adam(model.parameters(), lr=lr)
    elif (optimizer_name == "SGD"):
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        print("Name of optimizer should be one of { Adam, SGD }")

# do dotrenowania lr powinno być mniejsze od tego w pierwotnym uczeniu


def get_optimizer_for_model_unfreeze_layers(model, optimizer_name="Adam", lr=0.00001, momentum=0.9):
    if (optimizer_name == "Adam"):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif (optimizer_name == "SGD"):
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)
    else:
        print("Name of optimizer should be one of { Adam, SGD }")


def get_CIFAR10_dataset(train_dataset=True, perform_transform=True):
    return datasets.CIFAR10(
        root=dataset_root,
        train=train_dataset,
        download=True,
        transform=transform if perform_transform else None
    )


def save_model(model, model_name, classes):
    torch.save({
        "model_state": model.state_dict(),
        "num_classes": len(classes),
        "class_names": classes
    }, models_root + "/" + model_name + ".pth")


def read_model(model_name, model_type):
    model_info = torch.load(models_root + "/" + model_name + ".pth")
    if (model_type == "VGG16"):
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, model_info["num_classes"])
    elif (model_type == "AlexNet"):
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, model_info["num_classes"])
    elif (model_type == "GoogleNet"):
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(1024, model_info["num_classes"])
    else:
        print("Type of model shuold be one of { VGG16, AlexNet, GoogleNet }")
    model.load_state_dict(model_info["model_state"])
    classes = model_info["class_names"]
    return model, classes


def freeze_backbone(model, model_type):
    if model_type in ["VGG16", "AlexNet"]:
        for param in model.features.parameters():
            param.requires_grad = False

    elif model_type == "GoogleNet":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    else:
        print("Type of model shuold be one of { VGG16, AlexNet, GoogleNet }")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_last_layers(model, model_type):
    if model_type in ["VGG16", "AlexNet"]:
        for param in model.features[-4:].parameters():
            param.requires_grad = True

    elif model_type == "GoogleNet":
        for name, param in model.named_parameters():
            if "inception5" in name or "fc" in name:
                param.requires_grad = True

    else:
        print("Type of model shuold be one of { VGG16, AlexNet, GoogleNet }")


class DatasetWrapper(Dataset):
    def __init__(
        self,
        dataset,
        indices=None,
        remove_class=None,
        remap_labels=False
    ):
        self.dataset = dataset

        # ---------------------------------
        # indeksy
        # ---------------------------------
        if indices is not None:
            self.indices = indices

        elif remove_class is not None:
            self.indices = [
                i for i, y in enumerate(dataset.targets)
                if y != remove_class
            ]

        else:
            self.indices = list(range(len(dataset)))

        # ---------------------------------
        # klasy
        # ---------------------------------
        if remove_class is not None:
            self.classes = [
                c for i, c in enumerate(dataset.classes)
                if i != remove_class
            ]
        else:
            self.classes = dataset.classes.copy()

        self.class_to_idx = {
            c: i for i, c in enumerate(self.classes)
        }

        # ---------------------------------
        # targety
        # ---------------------------------
        self.targets = []

        for i in self.indices:
            label = dataset.targets[i]

            if remove_class is not None and remap_labels:
                if label > remove_class:
                    label -= 1

            self.targets.append(label)

        self.remove_class = remove_class
        self.remap_labels = remap_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]

        if self.remove_class is not None and self.remap_labels:
            if label > self.remove_class:
                label -= 1

        return image, label


# =====================================================
# SPLIT DATASETU NA:
# 1. jedną klasę
# 2. resztę klas
# =====================================================

def split_dataset(dataset, class_id, remap_rest=True):
    # ------------------------------
    # tylko jedna klasa
    # ------------------------------
    class_indices = [
        i for i, y in enumerate(dataset.targets)
        if y == class_id
    ]

    one_class_dataset = DatasetWrapper(
        dataset,
        indices=class_indices,
        remap_labels=False
    )

    one_class_dataset.classes = [dataset.classes[class_id]]
    one_class_dataset.class_to_idx = {
        dataset.classes[class_id]: 0
    }
    one_class_dataset.targets = [0] * len(one_class_dataset)

    # ------------------------------
    # reszta klas
    # ------------------------------
    rest_dataset = DatasetWrapper(
        dataset,
        remove_class=class_id,
        remap_labels=remap_rest
    )

    return one_class_dataset, rest_dataset


def set_dropout(model, model_type, dropout_value=None):
    # Dodane do eksperymentów z dropoutem
    if model_type in ["VGG16", "AlexNet"]:
        if dropout_value is None:
            model.classifier[2] = nn.Identity()
            model.classifier[5] = nn.Identity()
        else:
            model.classifier[2] = nn.Dropout(p=dropout_value)
            model.classifier[5] = nn.Dropout(p=dropout_value)

    else:
        print("Dropout setting is supported only for { VGG16, AlexNet }")

    return model
