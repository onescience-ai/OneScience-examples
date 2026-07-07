import torch


class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        if d <= 0 or p <= 0:
            raise ValueError("d and p must be positive")

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )

        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            loss = diff_norms / y_norms
            return torch.mean(loss) if self.size_average else torch.sum(loss)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
