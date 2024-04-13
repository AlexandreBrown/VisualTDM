import torch.nn as nn
import torch


class MiniResNet3(nn.Module):
    def __init__(self, in_channels: int, fc1_in_features: int, fc1_out_features: int, out_dim: int):
        super().__init__()
        
        self.stage_1 = self.create_stage_1(in_channels=in_channels)
        
        self.stage_2_conv_block = self.get_convolutional_block(in_channels=64, filters=[64, 64, 256], s=1)
        self.stage_2_id_block_1 = self.get_identity_block(in_channels=256, filters=[64, 64, 256])
        self.stage_2_id_block_2 = self.get_identity_block(in_channels=256, filters=[64, 64, 256])
        
        self.stage_3_conv_block = self.get_convolutional_block(in_channels=256, filters=[128, 128, 512], s=2)
        self.stage_3_id_block_1 = self.get_identity_block(in_channels=512, filters=[128, 128, 512])
        self.stage_3_id_block_2 = self.get_identity_block(in_channels=512, filters=[128, 128, 512])
        
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=fc1_out_features)
        self.fc2 = nn.Linear(in_features=fc1_out_features, out_features=out_dim)
    
    def create_stage_1(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
    
    def get_convolutional_block(self, in_channels, filters, s=2) -> nn.ModuleList:
        
        block = nn.ModuleList()
        
        F1, F2, F3 = filters
        
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=(1,1), stride=(s,s), padding='valid'))
        block.append(nn.BatchNorm2d(num_features=F1))
        block.append(nn.ReLU())
        
        block.append(nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(3,3), stride=(1,1), padding='same'))
        block.append(nn.BatchNorm2d(num_features=F2))
        block.append(nn.ReLU())
        
        block.append(nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=(1,1), stride=(1,1), padding='valid'))
        block.append(nn.BatchNorm2d(num_features=F3))
        
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=F3, kernel_size=(1,1), stride=(s,s), padding='valid'))
        block.append(nn.BatchNorm2d(num_features=F3))
        
        block.append(nn.ReLU())
        
        return block        

    def get_identity_block(self, in_channels, filters):
        
        block = nn.ModuleList()
        
        F1, F2, F3 = filters
        
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=F1, kernel_size=(1,1), stride=(1,1), padding='valid'))
        block.append(nn.BatchNorm2d(num_features=F1))
        block.append(nn.ReLU())
        
        block.append(nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(3,3), stride=(1,1), padding='same'))
        block.append(nn.BatchNorm2d(num_features=F2))
        block.append(nn.ReLU())
        
        block.append(nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=(1,1), stride=(1,1), padding='valid'))
        block.append(nn.BatchNorm2d(num_features=F3))
        
        block.append(nn.ReLU())
        
        return block
    
    def forward(self, x: torch.Tensor, additional_fc_features: torch.Tensor) -> torch.Tensor:
        x = self.stage_1(x)

        x = self.apply_conv_block(x, self.stage_2_conv_block)
        x = self.apply_identity_block(x, self.stage_2_id_block_1)
        x = self.apply_identity_block(x, self.stage_2_id_block_2)

        x = self.apply_conv_block(x, self.stage_3_conv_block)
        x = self.apply_identity_block(x, self.stage_3_id_block_1)
        x = self.apply_identity_block(x, self.stage_3_id_block_2)

        x =  nn.functional.adaptive_avg_pool2d(x, output_size=1)
        
        x = torch.flatten(x, start_dim=1)
        
        x = torch.cat([x, additional_fc_features], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
    def apply_conv_block(self, X, conv_block):
        X_shortcut = X
        for component in conv_block[:-3]:
            X = component(X)
        X_shortcut = conv_block[-3](X_shortcut)
        X_shortcut = conv_block[-2](X_shortcut)
        X = torch.add(X, X_shortcut)
        X = conv_block[-1](X)
        return X
    
    def apply_identity_block(self, X, block):
        X_shortcut = X
        for component in block[:-1]:
            X = component(X)
        X = torch.add(X, X_shortcut)
        X = block[-1](X)
        return X

if __name__ == '__main__':
    x = torch.randn(10, 3, 128, 128)
    goal = torch.randn(10, 256)
    tau = torch.randn(10, 1)
    model = MiniResNet3(in_channels=x.shape[1], goal_dim=goal.shape[1], fc1_out_features=256, out_dim=goal.shape[1])
    print(model(x, goal, tau).shape)
