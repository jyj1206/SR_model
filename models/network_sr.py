import torch
import torch.nn as nn

class SRB(nn.Module):
    def __init__(self, embed_dim, act='relu', resi=True):
        super().__init__()
        
        self.resi = resi
        
        if act == 'relu':
            activation = nn.ReLU()
        elif act == 'leakyrelu':
            activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {act}")

        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            activation,  
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )
        
        if self.resi:
            self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.resi:
            return x + self.res_scale * self.body(x)
        else:
            return self.body(x)


class SuperResolution(nn.Module):
    def __init__(self, scale = 2, num_layer = 10, embed_dim=64, act = 'relu', resi = True):
        super().__init__()
        
        self.resi = resi
        
        if act == 'relu':
            activation = nn.ReLU()
        elif act == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise ValueError(f"Unsupported activation: {act}")
        

        self.first_conv = nn.Conv2d(3, embed_dim, 3, 1, 1)
        
        self.layers = nn.ModuleList([
                SRB(embed_dim, act=act, resi=self.resi) for _ in range(num_layer)
            ])
        
        if self.resi:
            self.res_scale = nn.Parameter(torch.tensor(1.0))
            self.before_resi = nn.Conv2d(embed_dim, embed_dim, 3 , 1, 1)
        
        self.before_upsample = nn.Conv2d(embed_dim, (scale**2) * embed_dim, 3, 1, 1)
        
        self.upsample = nn.PixelShuffle(scale)

        self.final_conv = nn.Conv2d(embed_dim, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.first_conv(x)
        
        x_res = x
            
        for layer in self.layers:
            x = layer(x)
        
        if self.resi:
            x = self.before_resi(x)
            x = x_res + self.res_scale * x
            
        x = self.before_upsample(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        
        return x

