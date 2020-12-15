import torch


class ColorNet(torch.nn.Module):
    def __init__(self,num_classes,lite=False):
        super(ColorNet,self).__init__()
        self.lite = lite
        self.conv1_b1_1 = torch.nn.Conv2d(in_channels=3,
                                     out_channels=48,
                                     kernel_size=11,
                                     stride=4)
        self.activation1_b1_1 = torch.nn.ReLU()
        self.conv1_b2_1 = torch.nn.Conv2d(in_channels=3,
                                        out_channels=48,
                                        kernel_size=11,
                                        stride=4)
        self.activation1_b2_1 = torch.nn.ReLU()
        self.maxpool1_b1_1 = torch.nn.MaxPool2d(stride=2,kernel_size=3)
        self.maxpool1_b2_1 = torch.nn.MaxPool2d(stride=2,kernel_size=3)
        self.batchnorm1_b1_1 = torch.nn.BatchNorm2d(num_features=48, momentum=0.03, eps=1E-4)
        self.batchnorm1_b2_1 = torch.nn.BatchNorm2d(num_features=48, momentum=0.03, eps=1E-4)






        self.conv2_b1_1 = torch.nn.Conv2d(in_channels=48,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1)
        self.activation2_b1_1 = torch.nn.ReLU()

        self.conv2_b1_2 = torch.nn.Conv2d(in_channels=48,
                                          out_channels=64,
                                          kernel_size=3,
                                          stride=1)
        self.activation2_b1_2 = torch.nn.ReLU()

        self.conv2_b2_1 = torch.nn.Conv2d(in_channels=48,
                                          out_channels=64,
                                          kernel_size=3,
                                          stride=1)
        self.activation2_b2_1 = torch.nn.ReLU()

        self.conv2_b2_2 = torch.nn.Conv2d(in_channels=48,
                                          out_channels=64,
                                          kernel_size=3,
                                          stride=1)
        self.activation2_b2_2 = torch.nn.ReLU()

        self.maxpool2_b1_1 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.maxpool2_b1_2 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.maxpool2_b2_1 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.maxpool2_b2_2 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.batchnorm2_b1_1 = torch.nn.BatchNorm2d(num_features=64, momentum=0.03, eps=1E-4)
        self.batchnorm2_b1_2 = torch.nn.BatchNorm2d(num_features=64, momentum=0.03, eps=1E-4)
        self.batchnorm2_b2_1 = torch.nn.BatchNorm2d(num_features=64, momentum=0.03, eps=1E-4)
        self.batchnorm2_b2_2 = torch.nn.BatchNorm2d(num_features=64, momentum=0.03, eps=1E-4)



        self.conv3_b1_1 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=1,padding=1)
        self.activation3_b1_1 = torch.nn.ReLU()

        self.conv3_b1_2 = torch.nn.Conv2d(in_channels=64,
                                          out_channels=96,
                                          kernel_size=3,
                                          stride=1, padding=1)
        self.activation3_b1_2 = torch.nn.ReLU()
        self.conv3_b2_1 = torch.nn.Conv2d(in_channels=64,
                                          out_channels=96,
                                          kernel_size=3,
                                          stride=1, padding=1)
        self.activation3_b2_1 = torch.nn.ReLU()

        self.conv3_b2_2 = torch.nn.Conv2d(in_channels=64,
                                          out_channels=96,
                                          kernel_size=3,
                                          stride=1, padding=1)
        self.activation3_b2_2 = torch.nn.ReLU()

        self.conv4_b1_1 = torch.nn.Conv2d(in_channels=192,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=1,padding=1)
        self.activation4_b1_1 = torch.nn.ReLU()
        self.conv4_b1_2 = torch.nn.Conv2d(in_channels=192,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=1, padding=1)
        self.activation4_b1_2 = torch.nn.ReLU()

        self.conv4_b2_1 = torch.nn.Conv2d(in_channels=192,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=1, padding=1)
        self.activation4_b2_1 = torch.nn.ReLU()

        self.conv4_b2_2 = torch.nn.Conv2d(in_channels=192,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=1, padding=1)
        self.activation4_b2_2 = torch.nn.ReLU()




        self.conv5_b1_1 = torch.nn.Conv2d(in_channels=96,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1)
        self.activation5_b1_1 = torch.nn.ReLU()
        self.conv5_b1_2 = torch.nn.Conv2d(in_channels=96,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1)
        self.activation5_b1_2 = torch.nn.ReLU()

        self.conv5_b2_1 = torch.nn.Conv2d(in_channels=96,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1)
        self.activation5_b2_1 = torch.nn.ReLU()

        self.conv5_b2_2 = torch.nn.Conv2d(in_channels=96,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1)
        self.activation5_b2_2 = torch.nn.ReLU()

        self.maxpool5_b1_1 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.maxpool5_b1_2 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.maxpool5_b2_1 = torch.nn.MaxPool2d(stride=2, kernel_size=3)
        self.maxpool5_b2_2 = torch.nn.MaxPool2d(stride=2, kernel_size=3)


        if not lite:
            self.fc1 = torch.nn.Linear(in_features=4096,out_features=4096)
            self.activation_fc1 = torch.nn.ReLU()

            self.dropout1 = torch.nn.Dropout(p=0.2)

            self.fc2 = torch.nn.Linear(in_features=4096,out_features=4096)
            self.activation_fc2 = torch.nn.ReLU()

            self.dropout2 = torch.nn.Dropout(p=0.2)

            self.fc3 = torch.nn.Linear(in_features=4096,out_features=num_classes)
        else:
            self.fc1 = torch.nn.Linear(in_features=4096, out_features=512)
            self.activation_fc1 = torch.nn.ReLU()

            self.dropout1 = torch.nn.Dropout(p=0.2)

            self.fc2 = torch.nn.Linear(in_features=512, out_features=num_classes)


    def forward(self,x):
        x1_b1_1 = self.conv1_b1_1(x)
        x1_b1_1 = self.activation1_b1_1(x1_b1_1)
        x1_b2_1 = self.conv1_b2_1(x)
        x1_b2_1 = self.activation1_b2_1(x1_b2_1)
        x1_b1_1 = self.maxpool2_b1_1(x1_b1_1)
        x1_b2_1 = self.maxpool1_b2_1(x1_b2_1)
        x1_b1_1 = self.batchnorm1_b1_1(x1_b1_1)
        x1_b2_1 = self.batchnorm1_b2_1(x1_b2_1)

        x2_b1_1 = self.conv2_b1_1(x1_b1_1)
        x2_b1_1 = self.activation2_b1_1(x2_b1_1)
        x2_b1_2 = self.conv2_b1_2(x1_b1_1)
        x2_b1_2 = self.activation2_b1_2(x2_b1_2)
        x2_b2_1 = self.conv2_b2_1(x1_b2_1)
        x2_b2_1 = self.activation2_b2_1(x2_b2_1)
        x2_b2_2 = self.conv2_b2_2(x1_b2_1)
        x2_b2_2 = self.activation2_b2_2(x2_b2_2)
        x2_b1_1 = self.maxpool2_b1_1(x2_b1_1)
        x2_b1_2 = self.maxpool2_b1_2(x2_b1_2)
        x2_b2_1 = self.maxpool2_b2_1(x2_b2_1)
        x2_b2_2 = self.maxpool2_b2_2(x2_b2_2)
        x2_b1_1 = self.batchnorm2_b1_1(x2_b1_1)
        x2_b1_2 = self.batchnorm2_b1_2(x2_b1_2)
        x2_b2_1 = self.batchnorm2_b2_1(x2_b2_1)
        x2_b2_2 = self.batchnorm2_b2_2(x2_b2_2)

        x3_b1_1 = self.conv3_b1_1(x2_b1_1)
        x3_b1_1 = self.activation3_b1_1(x3_b1_1)
        x3_b1_2 = self.conv3_b1_2(x2_b1_2)
        x3_b1_2 = self.activation3_b1_2(x3_b1_2)
        x3_b2_1 = self.conv3_b2_1(x2_b2_1)
        x3_b2_1 = self.activation3_b2_1(x3_b2_1)
        x3_b2_2 = self.conv3_b2_2(x2_b2_2)
        x3_b2_2 = self.activation3_b2_2(x3_b2_2)
        x3_b1 =  torch.cat((x3_b1_1,x3_b1_2),dim=1)
        x3_b2 = torch.cat((x3_b2_1,x3_b2_2),dim=1)

        x4_b1_1 = self.conv4_b1_1(x3_b1)
        x4_b1_1 = self.activation4_b1_1(x4_b1_1)
        x4_b1_2 = self.conv4_b1_2(x3_b1)
        x4_b1_2 = self.activation4_b1_2(x4_b1_2)
        x4_b2_1 = self.conv4_b2_1(x3_b2)
        x4_b2_1 = self.activation4_b2_1(x4_b2_1)
        x4_b2_2 = self.conv4_b2_2(x3_b2)
        x4_b2_2 = self.activation4_b2_2(x4_b2_2)


        x5_b1_1 = self.conv5_b1_1(x4_b1_1)
        x5_b1_1 = self.activation5_b1_1(x5_b1_1)

        x5_b1_2 = self.conv5_b1_2(x4_b1_2)
        x5_b1_2 = self.activation5_b1_2(x5_b1_2)

        x5_b2_1 = self.conv5_b2_1(x4_b2_1)
        x5_b2_1 = self.activation5_b2_1(x5_b2_1)

        x5_b2_2 = self.conv5_b2_2(x4_b2_2)
        x5_b2_2 = self.activation5_b2_2(x5_b2_2)

        x5_b1_1 = self.maxpool5_b1_1(x5_b1_1)
        x5_b1_2 = self.maxpool5_b1_2(x5_b1_2)
        x5_b2_1 = self.maxpool5_b2_1(x5_b2_1)
        x5_b2_2 = self.maxpool5_b2_2(x5_b2_2)

        xfc = torch.cat((x5_b1_1,x5_b1_2,x5_b2_1,x5_b2_2),dim=1)
        xfc = xfc.view(-1,4096)

        if not self.lite:
            xfc = self.fc1(xfc)
            xfc = self.activation_fc1(xfc)
            xfc = self.dropout1(xfc)
            xfc = self.fc2(xfc)
            xfc = self.activation_fc2(xfc)
            xfc = self.dropout2(xfc)
            xfc = self.fc3(xfc)
        else:
            xfc = self.fc1(xfc)
            xfc = self.activation_fc1(xfc)
            xfc = self.dropout1(xfc)
            xfc = self.fc2(xfc)
        return xfc









        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = torch.cat((x,x),dim=1)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.maxpool5(x)
        # x = torch.cat((x,x,x,x),dim=1)
        # x = x.view(-1, 4096)
        # x = self.fc1(x)
        # x = self.fc2(x)
        #
        # return x


# img = torch.rand((1,3,227,227),dtype=torch.float32)
# net = ColorNet(11)
# print(net(img).shape)
