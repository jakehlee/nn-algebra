import torch

# referencing "PyTorch: Custom nn Modules"
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout

class ConvShortout(torch.nn.Module):

    def __init__(self, p=0.01):
        # p here is the short probability
        super(ConvShortout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("shortout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            # I'm assuming that X is (N, C, H, W).

            # 1. Maxpool the entire channel
            maxes = torch.nn.functional.max_pool2d(X, kernel_size=X.size()[2:])
            # maxes will be (N, C, 1, 1)
            maxes2 = torch.cat([maxes] * X.size(2), 2)
            maxes3 = torch.cat([maxes] * X.size(3), 3)
            # maxes will be back to (N, C, H, W).

            # 2. Dropout the entire thing
            # We're going to implement this from scratch here
            dist = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask = dist.sample(X.size()).cuda()
            #mask = dist.sample(X.size())
            # mask will be (N, C, H, W)

            # 3. Replace dropped values with maxes
            # no rescaling for now...
            shorted = mask * X + (1-mask) * maxes3
            return shorted
            