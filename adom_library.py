import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "data")
MODELS_ROOT = os.path.join(BASE_DIR, "models")

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


def preprocess_pil_image(img):
    return transform(img)


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


def predict_topk_classes(model, image_tensor, classes, k=5):
    model = model.to(DEVICE)
    model.eval()

    image = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    topk_prob, topk_idx = torch.topk(probs, k)

    predictions = []

    for i in range(k):
        class_id = int(topk_idx[0][i].item())
        probability = float(topk_prob[0][i].item())

        predictions.append({
            "rank": i + 1,
            "class_id": class_id,
            "class_name": classes[class_id],
            "probability": probability,
            "probability_percent": probability * 100,
        })

    return predictions


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

    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
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
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%, Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")

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
        root=DATASET_ROOT,
        train=train_dataset,
        download=True,
        transform=transform if perform_transform else None
    )


def save_model(model, model_name, classes):
    file_path = MODELS_ROOT + "/" + model_name + ".pth"
    os.makedirs(MODELS_ROOT, exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "num_classes": len(classes),
        "class_names": classes
    }, file_path)


def load_models(model_specs):
    loaded = {}
    classes = None

    for display_name, spec in model_specs.items():
        model, model_classes = read_model(
            spec["model_name"], spec["model_type"])
        model = model.to(DEVICE)
        model.eval()

        loaded[display_name] = {
            "model": model,
            "model_type": spec["model_type"],
            "model_name": spec["model_name"],
        }

        if classes is None:
            classes = model_classes

    return loaded, classes


def read_model(model_name, model_type):
    model_info = torch.load(MODELS_ROOT + "/" + model_name + ".pth")
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


def evaluate_models_on_dataset(loaded_models, dataset, classes, batch_size=64):
    results = {}
    for display_name, info in loaded_models.items():
        results[display_name] = evaluate_model_on_dataset(
            info["model"], dataset, classes, batch_size=batch_size
        )
    return results


def evaluate_model_on_dataset(model, dataset, classes, batch_size=64, shuffle=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return evaluate_model_on_loader(model, loader, classes)


def evaluate_model_on_loader(model, data_loader, classes):
    model = model.to(DEVICE)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    per_class_total = {class_name: 0 for class_name in classes}
    per_class_correct = {class_name: 0 for class_name in classes}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # wybieramy klasę z najwyższym prawdopodobieństwem
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for true_label, predicted_label in zip(labels, predicted):
                true_id = int(true_label.item())
                predicted_id = int(predicted_label.item())
                class_name = classes[true_id]
                per_class_total[class_name] += 1
                if true_id == predicted_id:
                    per_class_correct[class_name] += 1

    per_class_accuracy = {}
    for class_name in classes:
        class_total = per_class_total[class_name]
        class_correct = per_class_correct[class_name]
        per_class_accuracy[class_name] = class_correct / \
            class_total if class_total > 0 else 0.0

    return {
        "accuracy_percent": 100 * correct / total if total > 0 else 0.0,
        "avg_loss": running_loss / len(data_loader) if len(data_loader) > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_accuracy,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }


def visualize_feature_maps(model, image_tensor, layer_indices=[0, 5, 10, 17, 24], max_maps=8):    
    model = model.to(DEVICE)
    model.eval()

    image = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        current_output = image

        for layer_index, layer in enumerate(model.features):
            current_output = layer(current_output)

            if layer_index in layer_indices:
                feature_maps = current_output.squeeze(0).detach().cpu()
                maps_to_show = min(max_maps, feature_maps.shape[0])

                plt.figure(figsize=(maps_to_show * 2, 2.5))
                plt.suptitle(
                    f"Feature maps - layer {layer_index}: {layer.__class__.__name__}",
                    fontsize=12
                )

                for i in range(maps_to_show):
                    plt.subplot(1, maps_to_show, i + 1)
                    plt.imshow(feature_maps[i], cmap="viridis")
                    plt.axis("off")

                plt.show()


def show_model_layers(model, only_features=True):
    if only_features and hasattr(model, "features"):
        print("=== model.features ===")
        for index, layer in enumerate(model.features):
            print(f"{index}: {layer}")

    else:
        print("=== cały model ===")
        for name, module in model.named_modules():
            print(f"{name}: {module}")
