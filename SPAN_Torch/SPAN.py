from json import decoder
from unicodedata import bidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
from torchinfo import summary
from torch.nn.init import zeros_, ones_, kaiming_uniform_

class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None
        
        if padding:
            if padding is True:
                padding = [int((k-1)/2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1,1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        
        x = self.point_conv(x)

        return x

class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super(MixDropout, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)
    
    def forward(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), kernel=3, activation=nn.ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(num_features=out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, inputs):
        pos = random.randint(1,3)

        x = self.conv1(inputs)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x

class DSCBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        #x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, dropout=0.4):
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=32, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=256, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=256, out_c=512, stride=(2,1), dropout=dropout),
            ConvBlock(in_c=512, out_c=512, stride=(2,1), dropout=dropout),
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout)
        ])
    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x


class PageDecoder(nn.Module):

    def __init__(self, out_cats):
        super(PageDecoder, self).__init__()
        self.dec_conv = nn.Conv2d(in_channels= 512, out_channels= out_cats, kernel_size=(5,5), padding=(2,2))
    
    def forward(self, inputs):
        x = self.dec_conv(inputs)
        return F.log_softmax(x, dim=1)

class RecurrentPageDecoder(nn.Module):

    def __init__(self, out_cats):
        super(RecurrentPageDecoder, self).__init__()
        self.dec_conv = nn.Conv2d(in_channels= 512, out_channels=out_cats, kernel_size=(5,5), padding=(2,2))
        self.dec_lstm = nn.LSTM(input_size=out_cats, hidden_size=256, bidirectional=True, batch_first=True)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)
    
    def forward(self, inputs):
        x = self.dec_conv(inputs)
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(0,2,1)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        return F.log_softmax(x, dim=2)

class TransformerPageDecoder(nn.Module):

    def __init__(self, out_cats):
        super(TransformerPageDecoder, self).__init__()
        self.dec_conv = nn.Conv2d(in_channels= 512, out_channels=out_cats, kernel_size=(5,5), padding=(2,2))
        self.projection = nn.Linear(in_features=out_cats, out_features=512)
        transf_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.dec_transf = nn.TransformerEncoder(transf_layer, num_layers=1)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)

    def forward(self, inputs):
        x = self.dec_conv(inputs)
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(0,2,1)
        x = self.projection(x)
        x = self.dec_transf(x)
        x = self.out_dense(x)
        return F.log_softmax(x, dim=2)

class LineDecoder(nn.Module):

    def __init__(self, out_cats):
        super(LineDecoder, self).__init__()

        self.ada_pool = nn.AdaptiveMaxPool2d((1,None))
        self.dec_conv = nn.Conv2d(in_channels=512, out_channels=out_cats, kernel_size=(1,1))
    
    def forward(self, inputs):
        x = self.ada_pool(inputs)
        x = self.dec_conv(x)
        x = torch.squeeze(x, dim=2)
        return F.log_softmax(x, dim=1)


class SPANPage(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None):
        super(SPANPage, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = PageDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class SPANPageRecurrent(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None):
        super(SPANPageRecurrent, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = RecurrentPageDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class SPANPageTransformer(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None):
        super(SPANPageTransformer, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)

        if pretrain_path != None:
            print(f"Loading weights from {pretrain_path}")
            self.encoder.load_state_dict(torch.load(pretrain_path), strict=True)

        self.decoder = TransformerPageDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class SPANStaves(nn.Module):
    def __init__(self, in_channels, out_cats):
        super(SPANStaves, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = LineDecoder(out_cats=out_cats)
    
    def save_encoder_weights(self, path):
        torch.save(self.encoder.state_dict(), path)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


def SPAN_Weight_Init(m):
    """
    Weights initialization for model training from scratch
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight is not None:
            kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            ones_(m.weight)
        if m.bias is not None:
            zeros_(m.bias)


def get_span_model(maxwidth, maxheight, in_channels, out_size, encoder_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPANPage(in_channels=in_channels, out_cats=out_size+1, pretrain_path=encoder_weights).to(device)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    
    return model, device

def get_span_recurrent_model(maxwidth, maxheight, in_channels, out_size, encoder_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPANPageRecurrent(in_channels=in_channels, out_cats=out_size+1, pretrain_path=encoder_weights).to(device)
    model.apply(SPAN_Weight_Init)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    
    return model, device

def get_span_transformer_model(maxwidth, maxheight, in_channels, out_size, encoder_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPANPageTransformer(in_channels=in_channels, out_cats=out_size+1, pretrain_path=encoder_weights).to(device)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])
    
    return model, device

def get_span_stave_model(maxwidth, maxheight, in_channels, out_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPANStaves(in_channels=in_channels, out_cats=out_size+1).to(device)
    summary(model, input_size=[(1,in_channels,maxheight,maxwidth)], dtypes=[torch.float])

    return model, device

if __name__ == "__main__":
    get_span_model(260,260, 1, 91)

