import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, code_size=1024):
        super(VAE, self).__init__()
        self.code_size = code_size

        # Define the encoder architecture
        self.relu = nn.ReLU()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()

        self.enc_conv_1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.enc_bn_1 = nn.BatchNorm2d(32)

        self.enc_conv_2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.enc_bn_2 = nn.BatchNorm2d(64)

        self.enc_conv_3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.enc_bn_3 = nn.BatchNorm2d(128)

        self.enc_conv_4 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.enc_bn_4 = nn.BatchNorm2d(256)
        self.enc_fc51 = nn.Linear(256, 1024, bias=False)
        self.enc_fc52 = nn.Linear(256, 1024, bias=False)

        # Define the Decoder architecture
        # NOTE: The decoder architecture should not be very high capacity
        # because this might lead to posterior collapse
        self.dec_fc = nn.Linear(1024, 14 * 14 * 256)
        self.dec_upsample = nn.Upsample(scale_factor=2)
        self.dec_conv_1 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.dec_bn_1 = nn.BatchNorm2d(128)

        self.dec_conv_2 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.dec_bn_2 = nn.BatchNorm2d(64)

        self.dec_conv_3 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.dec_bn_3 = nn.BatchNorm2d(32)

        self.dec_conv_4 = nn.Conv2d(32, 3, 3, padding=1, bias=False)
        self.dec_bn_4 = nn.BatchNorm2d(3)

    def encode(self, x):
        x = self.pool(self.relu(self.enc_bn_1(self.enc_conv_1(x))))
        x = self.pool(self.relu(self.enc_bn_2(self.enc_conv_2(x))))
        x = self.pool(self.relu(self.enc_bn_3(self.enc_conv_3(x))))
        x = self.pool(self.relu(self.enc_bn_4(self.enc_conv_4(x))))
        x = self.flatten(self.aap(x))
        return self.enc_fc51(x), self.enc_fc52(x)

    def decode(self, z):
        x = self.relu(self.dec_fc(z))
        x = x.reshape((-1, 256, 14, 14))
        x = self.dec_upsample(self.relu(self.dec_bn_1(self.dec_conv_1(x))))
        x = self.dec_upsample(self.relu(self.dec_bn_2(self.dec_conv_2(x))))
        x = self.dec_upsample(self.relu(self.dec_bn_3(self.dec_conv_3(x))))
        output_img = torch.sigmoid(self.dec_upsample(self.dec_bn_4(self.dec_conv_4(x))))
        return output_img

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return z, decoder_out, mu, logvar


if __name__ == '__main__':
    vae = VAE()
    input = torch.randn((1, 3, 224, 224))
    _, output, _, _ = vae(input)
    print(output.shape)
