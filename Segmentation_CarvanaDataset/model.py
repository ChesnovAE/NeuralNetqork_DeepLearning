import torch.nn as nn

#conv, bn, relu block
def block(in_planes, out_planes, upsample=False, kernel=3, stride=1, padding=1, output_padding=0):
    if upsample:
        conv2d = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding,  output_padding=output_padding)
    else:
        conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1)
    return nn.Sequential(conv2d,
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        
        #downsample
        self.block_d1 = block(3, 16, stride=2)
        #self.block_d2 = block(16, 16)
        self.block_d3 = block(16, 32)
        self.maxpool_1 = nn.MaxPool2d(2, return_indices=True)
        self.block_d4 = block(32, 32)
        self.block_d5 = block(32, 32)
        self.maxpool_2 = nn.MaxPool2d(2, return_indices=True)
        self.block_d6 = block(32, 64)
        
        #upsample
        self.block_u6 = block(64, 32)
        self.maxunpool_2 = nn.MaxUnpool2d(2)
        self.block_u7 = block(32, 32)
        self.block_u8 = block(32, 32)
        self.maxunpool_1 = nn.MaxUnpool2d(2)
        self.block_u9 = block(32, 16)
        #self.block_u10 = block(16, 16)
        self.block_u11 = block(16, 2, upsample=True, stride=2, output_padding=1)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        #downsample
        x = self.block_d1(x)
        #x = self.block_d2(x)
        x = self.block_d3(x)
        x, index_1 = self.maxpool_1(x)
        x = self.block_d4(x)
        x = self.block_d5(x)
        x, index_2 = self.maxpool_2(x)
        x = self.block_d6(x)
        
        #upsample
        x = self.block_u6(x)
        x = self.maxunpool_2(x, index_2)
        x = self.block_u7(x)
        x = self.block_u8(x)
        x = self.maxunpool_1(x, index_1)
        x = self.block_u9(x)
        #x = self.block_u10(x)
        x = self.block_u11(x)
        
        x = self.softmax(x)
        
        return x


