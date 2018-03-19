import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, transforms

from Model import Generator, Discriminator


class Trainer:
    def __init__(self, batch_size):
        self.BS = batch_size
        self.G = Generator().cuda()
        self.D = Discriminator().cuda()
        self.optim_D = optim.RMSprop(self.D.parameters(), lr=0.0001)
        self.optim_G = optim.RMSprop(self.G.parameters(), lr=0.0001)

    def _calc_gradient_penalty(self, real_img, fake_img):
        alpha = torch.rand(self.BS, 1).cuda()
        alpha = alpha.expand(real_img.size())

        interpolates = alpha * real_img + ((1 - alpha) * fake_img)
        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        d_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return d_gradient_penalty

    def _calc_loss(self, real_img, rand_vec):
        self.fake_img = self.G(rand_vec)
        real_logits = self.D(real_img)
        fake_logits = self.G(self.fake_img)
        d_gradient_penalty = self._calc_gradient_penalty(real_img, self.fake_img)
        self.D_loss = fake_logits.mean() + real_logits.mean() + 0.1*d_gradient_penalty
        self.G_loss = -fake_logits.mean()

    def optimize(self):
        self.D.zero_grad()
        self.G.zero_grad()
        fake_img = self.G(rand_vec)
        fake_logits = self.D(fake_img)
        real_logits = self.D(real_img)
        d_gradient_penalty = self._calc_gradient_penalty(real_img, self.fake_img)
        self.D_loss = fake_logits.mean() + real_logits.mean() + 0.1 * d_gradient_penalty
        self.G_loss = -fake_logits.mean()
        self.D_loss

