import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:  # 8
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):  # 4个中间节点
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)  # 14个平均操作

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):  # 对于每一个中间节点
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))  # 每个节点的多个平均操作求和，得到该点的输出
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)  # 合并4个节点的输出


class Encoder(nn.Module):

    def __init__(self, C, layers, steps=4, multiplier=4):
        super(Encoder, self).__init__()
        self._inC = C  # 4
        self._layers = layers  # 3
        self._steps = steps
        self._multiplier = multiplier
        C_curr = 8

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 8, 3, padding=0, bias=False),
            # nn.BatchNorm2d(8)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            # C_curr = C*(2**i)
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self._initialize_alphas()

    def new(self):
        model_new = Encoder(self._inC, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        return s0, s1

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))  # 14
        num_ops = len(PRIMITIVES)

        self.alphas = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
        self.alphas.requires_grad = True

        self._arch_parameters = [
            self.alphas
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_former = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            cell=gene_former, cell_concat=concat
        )
        return genotype


class Decoder(nn.Module):

    def __init__(self, C, layers, steps=4, multiplier=4):
        super(Decoder, self).__init__()
        self._inC = C  # 8
        self._layers = layers  # 2
        self._steps = steps
        self._multiplier = multiplier

        C_prev_prev, C_prev, C_curr = C*4, C*4, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            # C_curr = C//(2**i)
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.pad = nn.ReflectionPad2d(1)
        self.ConvLayer = nn.Conv2d(C_curr*multiplier, 1, 3, padding=0)
        # self.tanh = nn.Tanh()
        self._initialize_alphas()

    def new(self):
        model_new = Decoder(self._inC, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, s0, s1):
        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        output = self.pad(s1)
        output = self.ConvLayer(output)
        return output

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))  # 14
        num_ops = len(PRIMITIVES)

        self.alphas = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
        self.alphas.requires_grad = True

        self._arch_parameters = [
            self.alphas
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            cell=gene, cell_concat=concat
        )
        return genotype


