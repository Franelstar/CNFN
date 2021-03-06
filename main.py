import os
import numpy as np
import time
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import model as model
import util_fn as util
from data_read import read_movies_data

params = {
        "GPU_num": -1,

        "DATA": './data',
        "DATASET": "sub",

        "INPUT_RANGE": (-1,1),
        'MF_NUM': 2,
        'MF_FUNC': 'gaussmf',
        "WORD_DIM": 128,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        'B_NORM': True,
        'tr_center_en': True,
        'tr_sigma_en': False,

        "SAVE_DIR": 'save',
        'SAVE_DIR_NEW': None,
        "SIGMA": 0.15,
        "DROPOUT_PROB": 0.0,
        "EPOCH": 1,
        "BATCH_SIZE": 128,
        "LR": { "FUZZ": 0.0001,
                "CONV3": 0.1,
                 "CONV4": 0.1,
                 "CONV5": 0.1,
                "DEFUZZ": 0.0001, "FC": 0.001},  # FC 0.01
        "WD": 0,
        "EPS": 1e-20,
        'N_SHOW':1,
        'MST': [100, 120, 140],
        'GAMMA': 0.1,
        "SEED": 3,

        }

#torch.cuda.set_device(params['GPU_num'])

params['SAVE_DIR'] = os.path.join(params['SAVE_DIR'], time.strftime("%y%m%d_%H%M")+'_CNFN_ds-{}'.format(params['DATASET']))
if not os.path.isdir(params['SAVE_DIR']): os.makedirs(params['SAVE_DIR'])

txt_data = read_movies_data(params)

ex_cnt = 0
acc_list = []
lr_list = []
def CNFN_run(lr_fuzz, lr_conv3, lr_conv4, lr_conv5, lr_defuzz,lr_fc, sigma, drop, wd):
     global ex_cnt

     params['LR']['FUZZ'] = lr_fuzz
     params['LR']['DEFUZZ'] = lr_defuzz
     params['LR']['CONV3'] = lr_conv3
     params['LR']['CONV4'] = lr_conv4
     params['LR']['CONV5'] = lr_conv5
     params['LR']['FC'] = lr_fc
     params['SIGMA'] = round(sigma, 2)
     params['DR'] = round(drop, 2)
     params['WD'] = round(wd,7)

     ex_cnt += 1

     if params['SAVE_DIR_NEW'] is None:
         params['SAVE_DIR_NEW'] = params['SAVE_DIR']

     log_fname = os.path.join(params['SAVE_DIR_NEW'],
                              'It{}_'.format(ex_cnt) + datetime.datetime.now().strftime('%H%M'))
     logger, log_fh = util.setup_logger('logger', log_fname + '.log')

     logger.info('Model parameters: {}'.format(params))
     logger.info('Learning rate: {}'.format(params['LR']))
     print('Learning rate: {}'.format(params['LR']))

     data = {}
     data['sub_train'] = txt_data['X_train']
     data['sub_test'] = txt_data['X_test']
     data['y_train'] = txt_data['y_train']
     data['y_test'] = txt_data['y_test']
     data["vocab"] = sorted(list(set([w for sent in data["sub_train"] + data["sub_test"] for w in sent])))
     data["classes"] = sorted(list(set(data["y_train"])))
     data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
     data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

     params["MAX_SENT_LEN"] = max([len(sent) for sent in data["sub_train"] + data["sub_test"]])
     params["VOCAB_SIZE"] = len(data["vocab"])
     params["CLASS_SIZE"] = len(data["classes"])

     print("=" * 20 + "INFORMATION" + "=" * 20)
     print("DATASET:", params["DATASET"])
     print("VOCAB_SIZE:", params["VOCAB_SIZE"])
     print("MAX_SENT_LEN", params["MAX_SENT_LEN"])
     print("EPOCH:", params["EPOCH"])

     print("=" * 20 + "INFORMATION" + "=" * 20)
     util.log_params(logger, params)

     #cnfn = model.CNFN(params).cuda()
     cnfn = model.CNFN(params).cpu()

     pars = {}
     for name, param in cnfn.named_parameters():
         if param.requires_grad:
             pars[name] = param

     p_conv3 = []
     for key, value in pars.items():
         if ('conv3' in key) :
             p_conv3.append(value)
     p_conv4 = []
     for key, value in pars.items():
         if ('conv4' in key):
             p_conv4.append(value)
     p_conv5 = []
     for key, value in pars.items():
         if ('conv5' in key) :
             p_conv5.append(value)
     p_fuzz = []
     for key, value in pars.items():
         if ('fuzz' in key) :
             p_fuzz.append(value)
     p_defuz = []
     for key, value in pars.items():
         if ('defuz' in key):
             p_defuz.append(value)

     optimizer = optim.Adam([
         {'params': p_fuzz, 'lr': params['LR']['FUZZ'], 'weight_decay': params['WD']},
         {'params': p_conv3, 'lr': params['LR']['CONV3'], 'weight_decay': params['WD']},
         {'params': p_conv4, 'lr': params['LR']['CONV4'], 'weight_decay': params['WD']},
         {'params': p_conv5, 'lr': params['LR']['CONV5'], 'weight_decay': params['WD']},
         {'params': p_defuz, 'lr': params['LR']['DEFUZZ'], 'weight_decay': params['WD']},

     ],
         lr=params['LR']['FC'], weight_decay=params['WD'])
     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['MST'], gamma=params['GAMMA'])

     criterion = nn.CrossEntropyLoss(reduction='mean')

     sub_train, y_train = data["sub_train"], data["y_train"]
     for e in range(params["EPOCH"]):
         sum_loss = 0

         b_num = len(sub_train) // params["BATCH_SIZE"] + 1

         for i in range(0, len(sub_train), params["BATCH_SIZE"]):
             batch_range = min(params["BATCH_SIZE"], len(sub_train) - i)

             batch_x = [[data["word_to_idx"][w] for w in sent] +
                        [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                        for sent in sub_train[i:i + batch_range]]
             batch_y = [data["classes"].index(c) for c in y_train[i:i + batch_range]]

             batch_x = Variable(torch.LongTensor(batch_x), requires_grad=False)
             batch_y = Variable(torch.LongTensor(batch_y), requires_grad=False)

             optimizer.zero_grad()
             cnfn.train()
             pred = cnfn(batch_x)

             loss = criterion(pred, batch_y)
             loss.backward()

             cnfn.fconv.hi.fconv3['center'].grad = cnfn.fconv.hi.conv3.weight.grad
             cnfn.fconv.hi.fconv4['center'].grad = cnfn.fconv.hi.conv4.weight.grad
             cnfn.fconv.hi.fconv5['center'].grad = cnfn.fconv.hi.conv5.weight.grad
             cnfn.fconv.lo.fconv3['center'].grad = cnfn.fconv.lo.conv3.weight.grad
             cnfn.fconv.lo.fconv4['center'].grad = cnfn.fconv.lo.conv4.weight.grad
             cnfn.fconv.lo.fconv5['center'].grad = cnfn.fconv.lo.conv5.weight.grad
             optimizer.step()

             sum_loss += loss.item()

         scheduler.step()

         tr_loss = sum_loss/b_num

         logger.info(
             "epoch:{: d}/ loss:{:6.4f}".format(e + 1,tr_loss))

         if e % params['N_SHOW']==0:
             print("epoch:{: d}/ loss:{:6.4f}".format(e + 1,tr_loss))


     dev_acc = model.test(data, cnfn, params)
     print('Final test acc:{:.4f}'.format(dev_acc))
     print(type(cnfn))
     print(params)


     logger.removeHandler(log_fh)
     new_log_fname = log_fname + '_l{:.4f}_a{:.3f}.log'.format(tr_loss, dev_acc)
     os.rename(log_fname + '.log',new_log_fname)

     LR = params['LR']
     acc_list.append([ex_cnt, LR['FUZZ'], LR['DEFUZZ'], LR['CONV3'], LR['CONV4'], LR['CONV5'], LR['FC'],
                      params['SIGMA'], params['DR'], params['WD'],
                      tr_loss, dev_acc])

     acc_mat = np.array(acc_list)
     np.savetxt(os.path.join(params['SAVE_DIR_NEW'], "result.csv"), acc_mat, delimiter=',')
     avg_acc = np.average(acc_mat[:,-1])
     new_save_dname = '{}_it{}_aa{:.4f}'.format(params['SAVE_DIR'], ex_cnt, avg_acc)
     os.rename(params['SAVE_DIR_NEW'], new_save_dname)
     params['SAVE_DIR_NEW'] = new_save_dname
     print('SAVE_DIR:', new_save_dname)

     return dev_acc


lr_list = [
    {'FUZZ': 0.0130623, 'CONV3': 0.0026407, 'CONV4': 0.0013694, 'CONV5': 0.1948642, 'DEFUZZ': 0.0028055,
     'FC': 0.0011, 'SIGMA': 0.15, 'DROP': 0.5, 'WD': 8.86e-05}
    ]
rep = 1
rep_result = []
for r in range(rep):
    for i in range(len(lr_list)):
        cv_acc = CNFN_run(lr_fuzz=lr_list[i]['FUZZ'], lr_conv3=lr_list[i]['CONV3'],lr_conv4=lr_list[i]['CONV4'], lr_conv5=lr_list[i]['CONV5'],
                              lr_defuzz=lr_list[i]['DEFUZZ'], lr_fc = lr_list[i]['FC'],
                              sigma = lr_list[i]['SIGMA'], drop = lr_list[i]['DROP'], wd =  lr_list[i]['WD'])
