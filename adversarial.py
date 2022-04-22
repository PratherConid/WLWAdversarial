import torch
FUN = torch.nn.functional

def FGSM(model, x, target, eps, device='cuda'):
    for i in model.parameters():
        i.requires_grad = False
    x = x.to(device=device); target = target.to(device=device)
    x.requires_grad = True
    pred = model(x)
    if pred == target:
        return x.clone().detach()
    loss = FUN.nll_loss(pred, target)
    x.grad = None
    loss.backward()
    data_grad = x.grad
    adv_ex = torch.clamp(x - eps * torch.sign(data_grad), 0, 1)
    return adv_ex.detach()

def PGD(model, x, target, eps, iters=30, rlr=1, device='cuda'):
    for i in model.parameters():
        i.requires_grad = False
    x = x.to(device=device); target = target.to(device=device)
    adv_ex = x.clone().detach()
    for i in range(iters):
        adv_ex.grad = None
        adv_ex.requires_grad = True
        pred = model(adv_ex)
        if pred == target:
            return adv_ex.detach()
        loss = FUN.nll_loss(pred, target)
        loss.backward()
        adv_ex = torch.clamp(adv_ex - eps * rlr * adv_ex.grad / torch.max(torch.abs(adv_ex.grad)), x - eps, x + eps).detach()
    return adv_ex.detach()