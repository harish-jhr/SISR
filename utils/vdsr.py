class VDSR(nn.Module):
    def __init__(self, num_channels=3, depth=20, features=64):
        super(VDSR, self).__init__()
        layers = [nn.Conv2d(num_channels, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, num_channels, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.net(x)
        return torch.clamp(x + residual, 0.0, 1.0)  # x is already upsampled LR
