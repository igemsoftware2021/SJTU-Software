import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

def f1_loss(pred_a, true_a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_a  = -(F.relu(-pred_a+1)-1)

    true_a = true_a.unsqueeze(1)
    unfold = nn.Unfold(kernel_size=(3, 3), padding=1)
    true_a_tmp = unfold(true_a)
    w = torch.Tensor([0, 0.0, 0, 0.0, 1, 0.0, 0, 0.0, 0]).to(device)
    true_a_tmp = true_a_tmp.transpose(1, 2).matmul(w.view(w.size(0), -1)).transpose(1, 2)
    true_a = true_a_tmp.view(true_a.shape)
    true_a = true_a.squeeze(1)

    tp = pred_a*true_a
    tp = torch.sum(tp, (1,2))

    fp = pred_a*(1-true_a)
    fp = torch.sum(fp, (1,2))

    fn = (1-pred_a)*true_a
    fn = torch.sum(fn, (1,2))

    f1 = torch.div(2*tp, (2*tp + fp + fn))
    return 1-f1.mean()


def TagLoss(input, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    output = output.to(device)
    Crossloss = nn.CrossEntropyLoss()
    total_loss = torch.zeros(1).to(device)
    for i in range(input.shape[0]):
        pred = input[i]
        target = output[i]
        loss = Crossloss(pred, target)
        total_loss += loss
    return total_loss/input.shape[0]

def BCETagLoss(input, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    output = output.to(device)
    Crossloss = nn.BCEWithLogitsLoss()
    total_loss = torch.zeros(1).to(device)
    for i in range(input.shape[0]):
        pred = input[i]
        target = output[i]
        loss = Crossloss(pred, target)
        total_loss += loss
    return total_loss/input.shape[0]


def Tag_F1_loss(pred, label):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = pred.to(device)
    label = label.to(device)
    total_f1 = torch.zeros((1), requires_grad=True).to(device)
    for i in range(pred.shape[0]):
        f1 = torch.from_numpy(np.array(f1_score(pred[i].detach().cpu(), label[i].detach().cpu(), average='weighted')))
        f1 = f1.to(device)
        total_f1 = total_f1 + f1

        # confusion_matrix = torch.zeros((3, 3), requires_grad=True)
        # for t, p in zip(pred[i].view(-1), label[i].view(-1)):
        #
        #     confusion_matrix[t.long(), p.long()] = confusion_matrix[t.long(), p.long()] + 1
        #
        # recall0 = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]
        # recall1 = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
        # recall2 = (confusion_matrix.diag() / confusion_matrix.sum(1))[2]
        # precision0 = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]
        # precision1 = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
        # precision2 = (confusion_matrix.diag() / confusion_matrix.sum(0))[2]
        #
        # f11 = (precision0 * recall0 * 2) / (precision0 + recall0)
        # f12 = (precision1 * recall1 * 2) / (precision1 + recall1)
        # f13 = (precision2 * recall2 * 2) / (precision2 + recall2)
        #
        # f1 = torch.div((f11 + f12 + f13), 3)
        #
        # total_f1 = total_f1 + f1

        total_f1 = total_f1 / pred.shape[0]

    return 1-total_f1/pred.shape[0]