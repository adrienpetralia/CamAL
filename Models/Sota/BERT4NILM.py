import torch
import math
import numpy as np
import random

import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT4NILM(nn.Module):
    def __init__(self, window_size, c_in=1, c_out=1, dp_rate=0.1, C0=1, mask_prob=0.2, 
                 use_bert4nilm_postprocessing=False, cutoff=None, threshold=None, return_values='power'):
        super().__init__()
        
        self.cutoff    = cutoff if cutoff is not None else 10000 # Need to be provide
        self.threshold = threshold if threshold is not None else 0 # According to code implementation

        self.use_bert4nilm_postprocessing = use_bert4nilm_postprocessing
        self.return_values = return_values

        self.mask_prob = mask_prob

        self.original_len = window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = dp_rate

        self.hidden = 256
        self.heads = 2
        self.n_layers = 2
        
        self.output_size = c_out

        self.conv = nn.Conv1d(in_channels=c_in, out_channels=self.hidden, kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])

        self.deconv = nn.ConvTranspose1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()
        
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss(reduction='mean')
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')
        self.C0 = C0

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, x):
        # Input as B, C, L
        x_token = self.pool(self.conv(x)).permute(0, 2, 1)
        embedding = x_token + self.position(x)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        
        # Output as B, C, L
        if self.training:
            return x.permute(0, 2, 1)
        else:
            if self.use_bert4nilm_postprocessing:
                logits_energy = self.cutoff_energy(x * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                if self.return_values=='power':
                    return logits_energy.permute(0, 2, 1) / self.cutoff
                elif self.return_values=='states':
                    return logits_status.permute(0, 2, 1).double()
                else:
                    return logits_energy.permute(0, 2, 1) / self.cutoff, logits_status.permute(0, 2, 1).double()
            else:
                return x.permute(0, 2, 1)

    def forward_valid(self, x):
        # Input as B, C, L
        x_token = self.pool(self.conv(x)).permute(0, 2, 1)
        embedding = x_token + self.position(x)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        
        # Output as B, C, L
        return x.permute(0, 2, 1)
    
    def compute_status(self, data):
        """
        State activation based on threshold
        """
        status = (data >= self.threshold) * 1

        return status
    
    def cutoff_energy(self, data):
        """
        Apply cutoff and cuton
        """
        data[data < 5] = 0 # Remove very small value
        data[data > self.cutoff] = self.cutoff
        
        return data

    def mask_bert_one_instance(self, x, y, status):
        tokens = []
        labels = []
        on_offs = []

        for i in range(len(x)):
            prob = random.random()
            if prob < self.mask_prob:
                prob = random.random()
                if prob < 0.8:
                    tokens.append(-1)
                elif prob < 0.9:
                    tokens.append(np.random.normal())
                else:
                    tokens.append(x[i])

                labels.append(y[i])
                on_offs.append(status[i])
            else:
                tokens.append(x[i])
                labels.append(-1)
                on_offs.append(-1)

        return torch.tensor(np.array(tokens)), torch.tensor(np.array(labels)), torch.tensor(np.array(on_offs))

    def mask_bert_one_batch(self, batch):

        x_in, y_in, status_in = batch[0].clone(), batch[1].clone(), batch[2].clone()

        for i in range(x_in.shape[0]):
            x, y, status = self.mask_bert_one_instance(torch.squeeze(batch[0][i]).numpy(), torch.squeeze(batch[1][i]).numpy(), torch.squeeze(batch[2][i]).numpy())
            
            x_in[i, 0, :] = x
            y_in[i, 0, :] = y
            status_in[i, 0, :] = status

        return x_in, y_in, status_in
    
    def train_one_epoch(self, loader, optimizer, device='cuda'):
        """
        Train BERT for one epoch
        """
        self.train()
        total_loss = 0
        
        for batch in loader:

            seqs, labels, status = self.mask_bert_one_batch(batch)
            seqs, labels, status = Variable(seqs.float()).to(device), Variable(labels.float()).to(device), Variable(status.float()).to(device)
            
            # Forward model
            optimizer.zero_grad()
            logits = self.forward(seqs).permute(0, 2, 1)

            # Permute to meet BERT4NILM convention
            seqs, labels, status = seqs.permute(0, 2, 1), labels.permute(0, 2, 1), status.float().permute(0, 2, 1)
            #labels = labels / self.cutoff -> Already done in NILMDataset if using MaxScaling

            batch_shape = status.shape
            logits_energy = self.cutoff_energy(logits * self.cutoff).float()
            logits_status = self.compute_status(logits_energy).float()

            mask = (status >= 0)
            labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1]))
            logits_masked = torch.masked_select(logits, mask).view((-1, batch_shape[-1]))
            status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1]))
            logits_status_masked = torch.masked_select(logits_status, mask).view((-1, batch_shape[-1]))

            kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits_masked.contiguous().view(-1).double(), labels_masked.contiguous().view(-1).double())
            margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).double(), (status_masked * 2 - 1).contiguous().view(-1).double())
            loss = kl_loss + mse_loss + margin_loss
            
            on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), labels_on.contiguous().view(-1))
                loss += self.C0 * loss_l1_on / total_size
            
            loss.backward()
            optimizer.step() 
            total_loss += loss.item()
           
        total_loss = total_loss / len(loader)

        return total_loss

    def valid_one_epoch(self, loader, device='cuda'):
        """
        Valid BERT for one epoch
        """
        self.eval()
        relative_errors, absolute_errors = [], []
        acc_values, precision_values, recall_values, f1_values,  = [], [], [], []
        
        with torch.no_grad():
            for seqs, labels, status in loader:

                seqs, labels, status = Variable(seqs.float()).to(device), Variable(labels.float()).to(device), Variable(status.float()).to(device)
                
                # Forward model
                logits = self.forward(seqs).permute(0, 2, 1)

                # Permute to meet BERT4NILM convention
                seqs, labels, status = seqs.permute(0, 2, 1), labels.permute(0, 2, 1), status.float().permute(0, 2, 1)
                #labels = labels / self.cutoff -> Already done in NILMDataset if using MaxScaling

                logits_energy = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                rel_err, abs_err = relative_absolute_error(logits_energy.detach().cpu().numpy().squeeze(), labels.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach().cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())
            
        return_rel_err = np.array(relative_errors).mean(axis=0)
        #return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        #return_precision = np.array(precision_values).mean(axis=0)
        #return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)

        total_loss = return_rel_err + return_acc + return_f1

        return total_loss


def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute = [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max((label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)

def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)