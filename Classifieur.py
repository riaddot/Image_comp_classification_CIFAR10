import torch 
import torch.nn as nn
import torch.nn.functional as F

# classifier 4  480 220 100 50 24 = 80 acc 
# 480 300 220 150 100  = 81.7 acc
# 480 960 460 300 150  = 83% 

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=480, out_channels=960, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(960)
        self.conv2 = nn.Conv2d(in_channels=960, out_channels=560, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(560)
        self.conv4 = nn.Conv2d(in_channels=560 ,out_channels=300, kernel_size=2, stride=1, padding=1, padding_mode='reflect')
        self.bn4 = nn.BatchNorm2d(300)
        self.conv45 = nn.Conv2d(in_channels=300, out_channels=200, kernel_size=2, stride=1, padding=1, padding_mode='reflect')
        self.bn45 = nn.BatchNorm2d(200)
        self.conv5 = nn.Conv2d(in_channels=200, out_channels=50, kernel_size=2, stride=1, padding=1, padding_mode='reflect')
        self.bn5 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(50*11*11, 10)        

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))     
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn45(self.conv45(output)))
        output = F.relu(self.bn5(self.conv5(output)))  
        # torch.flatten(t)
        output = output.view(-1, 50*11*11)
        output = self.fc1(output)
        

        return output
    
if __name__ == "__main__":

    model = Network()
    y = torch.rand((1,480,4,4))
    total_params = sum(
    param.numel() for param in model.parameters())
    print(total_params)
    # z= model(y)
    # print(z.shape)