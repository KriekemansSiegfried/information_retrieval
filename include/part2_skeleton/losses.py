import torch


def biranking_loss(a, b, margin, eps=1e-8):
    # compute cosine similarity matrix of all pairs in a and b
    a_n, b_n = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    cossim = torch.mm(a_norm, b_norm.transpose(-2, -1))

    # similarity scores of corresponding pairs
    diag = torch.diag(cossim)

    # mask out similarity scores of corresponding pairs
    mask = torch.diag(torch.ones_like(diag))
    cossim = cossim * (mask == False).float()
    cossim = cossim - mask

    # similarity scores of the hard negatives
    hard_a, _ = torch.max(cossim, dim=-1)
    hard_b, _ = torch.max(cossim, dim=-2)

    # compute losses
    loss = torch.clamp(hard_a - diag + margin, min=0, max=1e6) + \
           torch.clamp(hard_b - diag + margin, min=0, max=1e6)

    num_sample = loss.size(0)
    return torch.sum(loss) / num_sample


def cross_modal_hashing_loss(S, F, G, B, gamma, eta):
    # enforce cross-modal similarity
    theta = 0.5 * torch.mm(F.t(), G)

    term1 = - torch.sum((S * theta - torch.log(1 + torch.exp(theta))))
    # enforce binary codes to preserve cross-modal similarity
    term2 = gamma * (torch.norm(B - F) ** 2 + torch.norm(B - G) ** 2)

    # enforce maximization of the information provided in each vector dimension
    term3 = eta * (torch.norm(torch.mm(F, torch.ones(F.size(-1), 1))) ** 2 +
                   torch.norm(torch.mm(G, torch.ones(G.size(-1), 1))) ** 2)
    loss = torch.sum(term1 + term2 + term3)
    num_sample = S.size(0)

    # print('loss {} => {}'.format(loss.item()/num_sample, (term1.item() / num_sample, term2.item() / num_sample)))
    print('loss {} => {}'.format(loss.item() / num_sample,
                                 (term1.item() / num_sample, term2.item() / num_sample, term3.item() / num_sample)))

    return loss / num_sample
