import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
import torch.nn.functional as F


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        op_names, indices = zip(*genotype.cell)
        concat = genotype.cell_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)




class Encoder(nn.Module):

    def __init__(self, C, layers, genotype):
        super(Encoder, self).__init__()
        self._inC = C  # 4
        self._layers = layers  # 3
        C_curr = 8

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 8, 3, padding=0, bias=False),
            # nn.BatchNorm2d(8)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        return s0, s1


class Decoder(nn.Module):

    def __init__(self, C, layers, genotype):
        super(Decoder, self).__init__()
        self._inC = C  # 8
        self._layers = layers  # 2
        C_prev_prev, C_prev, C_curr = C*4, C*4, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        self.pad = nn.ReflectionPad2d(1)
        self.ConvLayer = nn.Conv2d(C_curr*4, 1, 3, padding=0)

    def forward(self, s0, s1):
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        output = self.pad(s1)
        output = self.ConvLayer(output)
        return output



