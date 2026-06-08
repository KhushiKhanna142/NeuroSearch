# Operations definitions
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Depthwise separable conv: depthwise -> pointwise -> BN -> ReLU"""
    def __init__(self, C, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            # Depthwise
            nn.Conv2d(C, C, kernel_size, stride=stride,
                      padding=padding, groups=C, bias=False),
            # Pointwise
            nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """Pass-through — used for skip connections with stride=1"""
    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    """
    Halves spatial dimensions while preserving channel count.
    Used for skip connections that cross a reduction cell (stride=2).

    Applies two parallel conv1x1 ops on spatially offset slices of the
    input, then concatenates. Each branch outputs C//2 channels so the
    concatenated result has C channels total — matching the expected
    output of a reduction cell.
    """
    def __init__(self, C):
        super().__init__()
        assert C % 2 == 0, "FactorizedReduce requires even channel count"
        self.relu = nn.ReLU(inplace=False)
        # Two parallel 1x1 convs, each outputting C//2 channels
        self.conv_a = nn.Conv2d(C, C // 2, 1, stride=2, padding=0, bias=False)
        self.conv_b = nn.Conv2d(C, C // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        x = self.relu(x)
        # conv_a uses the normal grid; conv_b uses a 1-pixel offset grid
        # This ensures every input pixel is seen by exactly one branch
        a = self.conv_a(x)
        b = self.conv_b(x[:, :, 1:, 1:])  # offset by one pixel top-left
        # Pad b back to same spatial size as a if needed
        if a.shape != b.shape:
            b = nn.functional.pad(b, (0, a.shape[3] - b.shape[3],
                                       0, a.shape[2] - b.shape[2]))
        out = torch.cat([a, b], dim=1)
        return self.bn(out)


class Zero(nn.Module):
    """
    Kills the connection — outputs a zero tensor.
    stride is applied by slicing so the spatial dimensions still match
    what a real op at that stride would produce.
    """
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        # Subsample to match the spatial reduction a real op would do
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------

OPS = {
    'conv3x3': lambda C, stride: ConvBNReLU(C, C, 3, stride, 1),
    'conv1x1': lambda C, stride: ConvBNReLU(C, C, 1, stride, 0),
    'sep3x3':  lambda C, stride: SepConv(C, 3, stride, 1),
    'sep5x5':  lambda C, stride: SepConv(C, 5, stride, 2),
    'maxpool': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'avgpool': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1,
                                               count_include_pad=False),
    'skip':    lambda C, stride: Identity() if stride == 1
                                 else FactorizedReduce(C),
    'zero':    lambda C, stride: Zero(stride),
}

# Stable ordering for index <-> name conversion
IDX_TO_OP = list(OPS.keys())          # e.g. IDX_TO_OP[0] == 'conv3x3'
OP_TO_IDX = {op: i for i, op in enumerate(IDX_TO_OP)}
N_OPS = len(IDX_TO_OP)