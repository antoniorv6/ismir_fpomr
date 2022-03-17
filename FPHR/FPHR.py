import torch
from torch._C import device
from torchvision.models import resnet34, resnet18
import torch.nn as nn
from torchinfo import summary
from .PosEncoding import PositionalEncoding1D, PositionalEncoding2D
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(Encoder, self).__init__()
        resNet_module = resnet18(pretrained=False)
        
        modules = list(resNet_module.children())
        #conv_1 = modules[0]
        #conv_1 = nn.Conv2d(1, conv_1.out_channels,
        #    conv_1.kernel_size,
        #    conv_1.stride,
        #    conv_1.padding,
        #    bias=conv_1.bias)

        self.resnet = nn.Sequential(*modules[:-2])

        #self.resnet = nn.Sequential(*modules[:-2])
        self.conv_adapter = nn.Conv2d(in_channels=512, out_channels=d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding2D(d_model=d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        #self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.conv_adapter.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu"
        )

        if self.conv_adapter.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv_adapter.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.conv_adapter.bias, -bound, bound)
    
    def forward(self, inputs):
        x = self.resnet(inputs) # (B, 512, H, W)
        x = self.conv_adapter(x) # (B, D_MODEL, H, W)
        x = self.pos_encoder(x) # (B, D_MODEL, H, W)
        x = self.dropout(x)
        x = x.flatten(start_dim=2) # (B, D_MODEL, H*W)
        x = x.permute(2,0,1) # (H*W, B, D_MODEL)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_depth, vocab_size, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_size = d_model 
        self.pos_encoding = PositionalEncoding1D(d_model=d_model, dropout=dropout)
        trf_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                        nhead=num_heads, 
                                                        dim_feedforward=ff_depth, 
                                                        dropout=dropout, 
                                                        activation=F.gelu)
        self.trf_decoder = nn.TransformerDecoder(trf_decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.ff_mask = self.generate_square_subsequent_mask(1024)
        #self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-init_range, init_range)

    @staticmethod
    def generate_square_subsequent_mask(size: int):
        """Generate a triangular (size, size) mask."""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, inputs, encoder_out):
        B, T = inputs.shape

        y = inputs.permute(1,0) # (Sy, B)
        y = self.embedding(y.long()) * math.sqrt(self.embedding_size) # (Sy, B, D_MODEL)
        y = self.pos_encoding(y) # (Sy, B, D_MODEL)
        
        #y = self.dropout(y)
        Sy = y.shape[0]
        y_mask = self.ff_mask[:Sy, :Sy].type_as(encoder_out)
        output = self.trf_decoder(y, encoder_out, y_mask) # (Sy, B, D_MODEL)
        output = self.output(output) # (Sy, B, VOCAB_CLASSES)
        return output

        #B, T = inputs.shape
        #tgt_key_padding_mask = inputs == 0
        #tgt_mask = self.subsequent_mask(T).to(inputs.device)
        #
        #x = self.embedding(inputs.long()) * math.sqrt(self.embedding_size)
        #x = self.pos_encoding(x)
        #x = self.dropout(x)
        #x = self.trf_decoder(x, encoder_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        #x = self.output(x)
        #return x

class FPHR(nn.Module):
    def __init__(self, d_model, n_decoder_layers, n_decoder_heads, ff_decoder_depth, output_size):
        super(FPHR, self).__init__()
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(num_layers=n_decoder_layers, 
                               d_model=d_model,
                               num_heads=n_decoder_heads,
                               ff_depth= ff_decoder_depth,
                               vocab_size= output_size)

    def forward(self, inputs, last_pred):
        x = self.encoder(inputs)
        out = self.decoder(inputs = last_pred, encoder_out = x)
        return out
    
    def forward_encoder(self, inputs):
        x = self.encoder(inputs)
        return x
    
    def forward_decoder(self, inputs, memory):
        x = self.decoder(inputs=inputs, encoder_out=memory)
        return F.log_softmax(x, dim=-1)

def get_fphr_model(maxwidth, maxheight, maxlen, out_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FPHR(n_decoder_layers=1, d_model=256, n_decoder_heads=4, ff_decoder_depth=1024, output_size=out_size).to(device)

    summary(model, input_size=[(1,3,maxheight,maxwidth), (1,maxlen)], dtypes=[torch.float, torch.float, torch.float, torch.bool, torch.bool, torch.bool])
    
    return model, device

if __name__ == "__main__":
    get_fphr_model()
