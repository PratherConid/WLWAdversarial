import torch
import numpy as np

def train(dataloader, model, loss_fn, optimizer, device='cuda'):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device='cuda'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def gradloss(model, x, batchnorm):
    for i in model.parameters():
        i.requires_grad = True
    if batchnorm:
        first_derivative = torch.autograd.functional.jacobian(model, x, create_graph=True)
        sum_first = torch.sum(first_derivative * first_derivative) / x.shape[0]
    else:
        sum_first = torch.tensor(0, dtype=x.dtype).to(device=x.device)
        for i in range(x.shape[0]):
            first_derivative = torch.autograd.functional.jacobian(model, x[i].reshape((1,) + x.shape[1:]), create_graph=True)
            sum_first += torch.sum(first_derivative * first_derivative)
    return sum_first

def gradloss_train(dataloader, model, loss_fn, optimizer, mu=0.1, device='cuda', batchnorm=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        grad_loss = gradloss(model, X, batchnorm)
        tot_loss = loss + mu * grad_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  gradient loss: {grad_loss:>7f}  total loss: {tot_loss:>7f}  [{current:>5d}/{size:>5d}]")

def gradient_test(dataloader, model, loss_fn, gl_ratio=0.1, device='cuda', batchnorm=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, grad_loss, gl_cnt = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if np.random.uniform(0, 1) < gl_ratio:
                grad_loss += float(gradloss(model, X, batchnorm))
                gl_cnt += 1
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    grad_loss /= gl_cnt
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Avg Gradient loss: {grad_loss:>8f} \n")