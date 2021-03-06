import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import util_fn as util_fn


class mf:
    def gaussmf(x, mean, sigma):
        """
        Gaussian fuzzy membership function.
        """
        if 'torch' in str(type(x)):
            return torch.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
        else:
            return np.exp(-((x - mean)**2.) / (2 * sigma**2.))

    def sigmmf(x, c, a):
        """
        Sigmoid fuzzy membership function.
        """
        if 'torch' in str(type(x)):
            return 1/(1+torch.exp(-a*(x - c)))
        else:
            return 1/(1+np.exp(-a*(x - c)))

class FuzzyConv(nn.Sequential):
    def __init__(self,params, mf_pars):
        super(FuzzyConv, self).__init__()

        np.random.seed(params['SEED'])
        torch.manual_seed(params['SEED'])
        torch.manual_seed(params['SEED'])

        dtype = torch.FloatTensor
        dim = params['WORD_DIM']    # 128
        sent_len = params['MAX_SENT_LEN']

        ci = 1                # Channel in
        co = params["FILTER_NUM"][0]  # Channel out 100

        Ks = params["FILTERS"]  # 3,4,5 #Kernel size
        sigmaf = params['SIGMA']

        sigmac = params['SIGMA']
        self.bn = params['B_NORM']
        self.tr_center = params['tr_center_en']
        self.tr_sigma = params['tr_sigma_en']

        self.fuzz = nn.ParameterDict({
            'center': torch.nn.Parameter(
                torch.FloatTensor(sent_len, dim).uniform_(mf_pars['f_center_init_min'],
                                                          mf_pars['f_center_init_max']).type(dtype),
                                                    requires_grad=self.tr_center),
            'sigma': torch.nn.Parameter(torch.FloatTensor(sent_len, dim).fill_(sigmaf).type(dtype),
                                        requires_grad=self.tr_sigma),
            'min': torch.nn.Parameter(torch.tensor(mf_pars['f_center_min']).type(dtype), requires_grad=False),
            'max': torch.nn.Parameter(torch.tensor(mf_pars['f_center_max']).type(dtype), requires_grad=False),
        })

        self.fconv3 =  nn.ParameterDict({
            'center': torch.nn.Parameter(
                torch.FloatTensor(co, ci, Ks[0], dim).uniform_(mf_pars['conv_center_init_min'],
                                                         mf_pars['conv_center_init_max']).type(dtype),
                                                        requires_grad=self.tr_center),
            'sigma': torch.nn.Parameter(torch.FloatTensor(co, ci, Ks[0], dim).fill_(sigmac).type(dtype),
                                  requires_grad=self.tr_sigma),

            'min': torch.nn.Parameter(torch.tensor(mf_pars['conv_center_min']).type(dtype), requires_grad=False),
            'max': torch.nn.Parameter(torch.tensor(mf_pars['conv_center_max']).type(dtype), requires_grad=False),
            'cent': torch.nn.Parameter(torch.tensor(mf_pars['fixed_cent']).type(dtype), requires_grad=False),
             })
        self.fconv4 = nn.ParameterDict({
            'center': torch.nn.Parameter(
                torch.FloatTensor(co, ci, Ks[1], dim).uniform_(mf_pars['conv_center_init_min'],
                                                               mf_pars['conv_center_init_max']).type(dtype),
                requires_grad=self.tr_center),
            'sigma': torch.nn.Parameter(torch.FloatTensor(co, ci, Ks[1], dim).fill_(sigmac).type(dtype),
                                        requires_grad=self.tr_sigma),
            'min': torch.nn.Parameter(torch.tensor(mf_pars['conv_center_min']).type(dtype), requires_grad=False),
            'max': torch.nn.Parameter(torch.tensor(mf_pars['conv_center_max']).type(dtype), requires_grad=False),
            'cent': torch.nn.Parameter(torch.tensor(mf_pars['fixed_cent']).type(dtype), requires_grad=False),
        })

        self.fconv5 =  nn.ParameterDict({
             'center': torch.nn.Parameter(
                torch.FloatTensor(co, ci, Ks[2], dim).uniform_(mf_pars['conv_center_init_min'],
                                                               mf_pars['conv_center_init_max']).type(dtype),
                requires_grad=self.tr_center),
            'sigma': torch.nn.Parameter(torch.FloatTensor(co, ci, Ks[2], dim).fill_(sigmac).type(dtype),
                                        requires_grad=self.tr_sigma),
            'min': torch.nn.Parameter(torch.tensor(mf_pars['conv_center_min']).type(dtype), requires_grad=False),
            'max': torch.nn.Parameter(torch.tensor(mf_pars['conv_center_max']).type(dtype), requires_grad=False),
            'cent': torch.nn.Parameter(torch.tensor(mf_pars['fixed_cent']).type(dtype), requires_grad=False),
        })

        self.defuz =  nn.ParameterDict({
            'center': torch.nn.Parameter(
                torch.FloatTensor(len(Ks) * co).uniform_(mf_pars['df_center_init_min'],
                                                         mf_pars['df_center_init_max']).type(dtype),
                                                        requires_grad=True),
         })

        self.add_module('conv3', nn.Conv2d(ci, co, (Ks[0], dim)))
        if self.bn:
              self.add_module('bnorm13', nn.BatchNorm2d(co))

        self.add_module('conv4', nn.Conv2d(ci, co, (Ks[1], dim)))
        if self.bn:
              self.add_module('bnorm14', nn.BatchNorm2d(co))

        self.add_module('conv5', nn.Conv2d(ci, co, (Ks[2], dim)))
        if self.bn:
              self.add_module('bnorm15', nn.BatchNorm2d(co))

class CNFN(nn.Module):

    def __init__(self, model_pars):
        super(CNFN, self).__init__()
        self.GPU_num = model_pars['GPU_num']

        self.eps = model_pars['EPS']
        self.len = model_pars['MAX_SENT_LEN']
        Vocab = model_pars["VOCAB_SIZE"]  # 21109
        Dim = model_pars['WORD_DIM']  # 128
        self.mf_num = model_pars['MF_NUM']
        self.input_dim = model_pars['WORD_DIM']
        self.mf_func = model_pars['MF_FUNC']
        self.tr_center = model_pars['tr_center_en']
        self.tr_sigma = model_pars['tr_sigma_en']
        C = model_pars["CLASS_SIZE"]  # 2
        Ci = 1  # Channel in
        Co = model_pars["FILTER_NUM"][0]  # Channel out 100
        Ks = model_pars["FILTERS"]  # 3,4,5 #Kernel size




        self.mf_params = {
            'lo':   {
                    'f_center_init_min': 0.249, 'f_center_init_max': 0.251,
                    'f_center_min': 0.0, 'f_center_max': 0.5,
                    'conv_center_init_min': 0.0, 'conv_center_init_max': 0.01,
                    'conv_center_min': -0.75, 'conv_center_max': 0.25,
                    'df_center_init_min': 0.249, 'df_center_init_max': 0.251,
                    'fixed_cent': 0.0,
                    },

            'hi':   {
                    'f_center_init_min': 0.749, 'f_center_init_max': 0.751,
                    'f_center_min': 0.5, 'f_center_max': 1,

                    'conv_center_init_min': 0.99, 'conv_center_init_max': 1.0,
                    'conv_center_min': 0.75, 'conv_center_max': 1.75,

                    'df_center_init_min': 0.7499, 'df_center_init_max': 0.7501,

                    'fixed_cent': 1.0,
                    },
            }

        self.embed = nn.Embedding(Vocab + 2, Dim, padding_idx=Vocab + 1)

        lt_val = np.arange(Dim)
        lookup_tensor = torch.from_numpy(lt_val)
        embed = self.embed(Variable(lookup_tensor))

        self.embed_min = nn.Parameter(embed.min(),requires_grad=False)
        self.embed_max = nn.Parameter(embed.max(),requires_grad=False)

        self.fconv = nn.Sequential()
        self.fconv.add_module('lo', FuzzyConv(model_pars, self.mf_params['lo']))
        self.fconv.add_module('hi', FuzzyConv(model_pars, self.mf_params['hi']))

        self.dropout = nn.Dropout(model_pars['DR'])
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = x.squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)
        x = util_fn.scale_embed(x, self.embed_min, self.embed_max, self.GPU_num);  self.log_x_embed_scaled = x
        num = 0
        den = 0
        for mf_idx in range(len(self.fconv)):
            center_arr = self.fconv[mf_idx].fuzz.center.expand_as(x)
            sigma_arr = self.fconv[mf_idx].fuzz.sigma.expand_as(x)
            delta = x - center_arr
            Xf = (-((delta).pow(2.0)) / sigma_arr.pow(2.0)).exp()
            x_in = Xf.unsqueeze(1)  # (N,Ci,W,D)

            self.fconv[mf_idx].conv3.weight.data = (-((self.fconv[mf_idx].fconv3['cent'] - self.fconv[mf_idx].fconv3['center'].data).pow(2.0)) /
                                 (self.fconv[mf_idx].fconv3['sigma'].data.pow(2.0) + self.eps)).exp()
            self.fconv[mf_idx].conv4.weight.data = (-((self.fconv[mf_idx].fconv4['cent'] - self.fconv[mf_idx].fconv4['center'].data).pow(2.0)) /
                                 (self.fconv[mf_idx].fconv4['sigma'].data.pow(2.0) + self.eps)).exp()
            self.fconv[mf_idx].conv5.weight.data = (-((self.fconv[mf_idx].fconv5['cent'] - self.fconv[mf_idx].fconv5['center'].data).pow(2.0)) /
                                 (self.fconv[mf_idx].fconv5['sigma'].data.pow(2.0) + self.eps)).exp()

            x_3 = F.relu(self.fconv[mf_idx].bnorm13(self.fconv[mf_idx].conv3(x_in))).squeeze(3)
            x_3 = F.max_pool1d(x_3, x_3.size(2)).squeeze(2)

            x_4 = F.relu(self.fconv[mf_idx].bnorm14(self.fconv[mf_idx].conv4(x_in))).squeeze(3)
            x_4 = F.max_pool1d(x_4, x_4.size(2)).squeeze(2)

            x_5 = F.relu(self.fconv[mf_idx].bnorm15(self.fconv[mf_idx].conv5(x_in))).squeeze(3)
            x_5 = F.max_pool1d(x_5, x_5.size(2)).squeeze(2)

            x_conv = torch.cat((x_3, x_4, x_5), 1)  # (N,len(Ks)*Co)

            x_conv = self.dropout(x_conv)

            den += x_conv
            num += x_conv * self.fconv[mf_idx].defuz['center'].expand_as(x_conv)

        den += self.eps
        x_out = num / den

        logit = self.fc1(x_out)  # (N,C)
        return logit

def test(data, model, params):
    model.eval()

    x, y = data["sub_test"], data["y_test"]
    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    y = [data["classes"].index(c) for c in y]

    pred = []
    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)
        batch_x = x[i:i + batch_range]

        batch_x = Variable(torch.LongTensor(batch_x))
        pred += list(np.argmax(model(batch_x).cpu().data.numpy(), axis=1))

    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)])/len(y)
    return acc
