import torchvision
import torch.nn as nn


def get_model(model_type,num_classes):

    if model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        trainable_params = model.fc.parameters()

    elif model_type == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        trainable_params = model.classifier.parameters()

    elif model_type == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        trainable_params = model.classifier.parameters()

    elif model_type == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.00001, momentum=0.9)
        trainable_params = model.classifier.parameters()

    elif model_type == 'densenet':
        model = torchvision.models.densenet121(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        trainable_params = model.classifier.parameters()

    elif model_type == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # optimizer = optim.Adam(model.fc.parameters())
        trainable_params = model.fc.parameters()

    elif model_type == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        trainable_params = model.fc.parameters()

    elif model_type == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        # optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        trainable_params = model.classifier.parameters()

    else:
        raise("Incorrect model specified!")

    return model,trainable_params

