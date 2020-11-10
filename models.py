import torchvision
import torch.nn as nn
from custom_models import TwoLayerConv100RGB


def get_model(model_type,num_classes,pretrained=True):
    if "-" not in model_type:
        if model_type == 'resnet18':
            if pretrained:
                model = torchvision.models.resnet18(pretrained=pretrained)
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                trainable_params = model.fc.parameters()
            else:
                model = torchvision.models.resnet18(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                trainable_params = model.parameters()

        elif model_type == 'vgg16':
            if pretrained:
                model = torchvision.models.vgg16(pretrained=pretrained)
                for param in model.features.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                trainable_params = model.classifier.parameters()
            else:
                model = torchvision.models.vgg16(pretrained=pretrained)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                trainable_params = model.parameters()

        elif model_type == 'squeezenet':
            if pretrained:
                model = torchvision.models.squeezenet1_1(pretrained=pretrained)
                for param in model.features.parameters():
                    param.requires_grad = False
                model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
                trainable_params = model.classifier.parameters()
            else:
                model = torchvision.models.squeezenet1_1(pretrained=pretrained)
                model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1),
                                                stride=(1, 1))
                trainable_params = model.parameters()

        elif model_type == 'densenet':
            if pretrained:
                model = torchvision.models.densenet121(pretrained=pretrained)
                for param in model.features.parameters():
                    param.requires_grad = False
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                trainable_params = model.classifier.parameters()
            else:
                model = torchvision.models.densenet121(pretrained=pretrained)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                trainable_params = model.parameters()


        elif model_type == 'googlenet':
            if pretrained:
                model = torchvision.models.googlenet(pretrained=pretrained)
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                # optimizer = optim.Adam(model.fc.parameters())
                trainable_params = model.fc.parameters()
            else:
                model = torchvision.models.googlenet(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                trainable_params = model.parameters()

        elif model_type == 'shufflenet':
            if pretrained:
                model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                trainable_params = model.fc.parameters()
            else:
                model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                trainable_params = model.parameters()

        elif model_type == 'mobilenet':
            if pretrained:
                model = torchvision.models.mobilenet_v2(pretrained=pretrained)
                for param in model.features.parameters():
                    param.requires_grad = False
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                trainable_params = model.classifier.parameters()
            else:
                model = torchvision.models.mobilenet_v2(pretrained=pretrained)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                trainable_params = model.parameters()

    elif model_type.split("-")[0] == 'custom':
        if model_type.split("-")[1] == "2layer100RGB":
            model = TwoLayerConv100RGB()
            trainable_params = model.parameters()
        else:
            raise("Incorrect custom model specified!")
    else:
        raise("Incorrect model specified!")

    return model,trainable_params


# model = get_model('custom-2layer100RGB',2,False)[0]
# print(sum([x.numel() for x in model.parameters()]))