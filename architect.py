import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model_former, model_latter, args, mse_loss, ssim_loss):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model_former = model_former
        self.model_latter = model_latter
        self.mse_loss = mse_loss
        self.ssim_loss = ssim_loss
        para = [{'params': model_former.arch_parameters(), 'lr': args.arch_learning_rate},
                {'params': model_latter.arch_parameters(), 'lr': args.arch_learning_rate}]
        self.optimizer = torch.optim.Adam(para,
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, eta, network_optimizer):
        loss = self.model._loss(input)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta[0]))
        return unrolled_model

    def step(self, input_train, input_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, input_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid):
        en1, en2 = self.model_former(input_valid)  # ######
        output_valid = self.model_latter(en1, en2)
        ssim_loss_value = 0.
        pixel_loss_value = 0.
        for output, input in zip(output_valid, input_valid):
            output, input = torch.unsqueeze(output, 0), torch.unsqueeze(input, 0)
            pixel_loss_temp = self.mse_loss(input, output)
            ssim_loss_temp = self.ssim_loss(input, output, normalize=True, val_range=255)
            ssim_loss_value += (1 - ssim_loss_temp)
            pixel_loss_value += pixel_loss_temp
        ssim_loss_value /= len(output_valid)
        pixel_loss_value /= len(output_valid)

        total_loss = pixel_loss_value + 100*ssim_loss_value  # 加权？
        total_loss.backward()

    def _backward_step_unrolled(self, input_train, input_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta[0])
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()
        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length
        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.model._loss(input)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = self.model._loss(input)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
