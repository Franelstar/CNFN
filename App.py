import streamlit as st
import pandas as pd

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
from data_read_2 import read_movies_data

# streamlit run App.py 

col6_1, col6_2 = st.sidebar.beta_columns(2)
# GPU_num
gpu_num = col6_1.selectbox(
    'GPU',
    (-1, 0, 1, 2)
)

# Save dir
save_dir = col6_2.text_input('Save directory', 'save')

col5_1, col5_2 = st.sidebar.beta_columns(2)
# Dropout probability
dropout_prob = col5_1.slider('Dropout probability', 0.0, 1.0, 0.0)

# Sigma
sigma = col5_2.select_slider('Sigma', options=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], value=0.15)

col4_1, col4_2 = st.sidebar.beta_columns(2)
# Epoch
epoch = col4_1.slider('Epochs', 1, 500, 1)

# Batch size
batch_size = col4_2.select_slider('Batch size', options=[32, 64, 128, 256, 512], value=128)

col3_1, col3_2 = st.sidebar.beta_columns(2)
# Learning rate Fuzz
lr_fuzz = col3_1.select_slider('Learning rate Fuzz',
                                   options=[0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                                   value=0.0001)

# Learning rate DeFuzz
lr_defuzz = col3_2.select_slider('Learning rate DeFuzz',
                                     options=[0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                                     value=0.0001)

col2_1, col2_2 = st.sidebar.beta_columns(2)
# Learning rate Conv3
lr_conv3 = col2_1.select_slider('Learning rate Conv3', options=[0.001, 0.01, 0.1, 0.15, 0.2, 0.3], value=0.1)

# Learning rate Conv4
lr_conv4 = col2_2.select_slider('Learning rate Conv4', options=[0.001, 0.01, 0.1, 0.15, 0.2, 0.3], value=0.1)


col1, col2 = st.sidebar.beta_columns(2)
# Learning rate Conv5
lr_conv5 = col1.select_slider('Learning rate Conv5', options=[0.001, 0.01, 0.1, 0.15, 0.2, 0.3], value=0.1)

# Learning rate FC
lr_fc = col2.select_slider('Learning rate FC', options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1], value=0.001)

params = {
    "GPU_num": gpu_num,

    "DATA": './data',
    "DATASET": "sub",

    "INPUT_RANGE": (-1, 1),
    'MF_NUM': 2,
    'MF_FUNC': 'gaussmf',
    "WORD_DIM": 128,
    "FILTERS": [3, 4, 5],
    "FILTER_NUM": [100, 100, 100],
    'B_NORM': True,
    'tr_center_en': True,
    'tr_sigma_en': False,

    "SAVE_DIR": save_dir,
    'SAVE_DIR_NEW': None,
    "SIGMA": sigma,
    "DROPOUT_PROB": dropout_prob,
    "EPOCH": epoch,
    "BATCH_SIZE": batch_size,
    "LR": {"FUZZ": lr_fuzz,
           "CONV3": lr_conv3,
           "CONV4": lr_conv4,
           "CONV5": lr_conv5,
           "DEFUZZ": lr_defuzz,
           "FC": lr_fc},  # FC 0.01
    "WD": 0,
    "EPS": 1e-20,
    'N_SHOW': 1,
    'MST': [100, 120, 140],
    'GAMMA': 0.1,
    "SEED": 3,
}

st.title('CNFN')

torch.cuda.set_device(params['GPU_num'])

params['SAVE_DIR'] = os.path.join(params['SAVE_DIR'],
                                  time.strftime("%y%m%d_%H%M") + '_CNFN_ds-{}'.format(params['DATASET']))
if not os.path.isdir(params['SAVE_DIR']): os.makedirs(params['SAVE_DIR'])

ex_cnt = 0
acc_list = []

lr_list = [
    {'FUZZ': 0.0130623, 'CONV3': 0.0026407, 'CONV4': 0.0013694, 'CONV5': 0.1948642, 'DEFUZZ': 0.0028055,
     'FC': 0.0011, 'SIGMA': 0.15, 'DROP': 0.5, 'WD': 8.86e-05}
]

# st.write(params)

st.subheader("Liste des paramètres")

params_1 = pd.DataFrame(
    np.array([[gpu_num, save_dir, dropout_prob, sigma, epoch, batch_size]]),
    columns=['GPU', 'Save Dir', 'Dropout prob', 'Sigma', 'Epochs', 'Batch size'])

st.dataframe(params_1)

params_2 = pd.DataFrame(
    np.array([[lr_fuzz, lr_defuzz, lr_conv3, lr_conv4, lr_conv5, lr_fc]]),
    columns=['LR Fuzz', 'LR DeFuzz', 'LR conv3', 'LR Conv4', 'LR Conv5', 'LR FC'])

st.dataframe(params_2)

st.sidebar.subheader("Données d'entrainement")
col10_1, col10_2, col10_3, col10_4 = st.sidebar.beta_columns(4)
col11_1, col11_2, col11_3, _ = st.sidebar.beta_columns(4)

bmi = col10_1.checkbox('BMI', value=True)
chi = col10_2.checkbox('CHI', value=True)
cra = col10_3.checkbox('CRA', value=False)
dep = col10_4.checkbox('DEP', value=False)
fne = col11_1.checkbox('FNE', value=True)
gla = col11_2.checkbox('GLA', value=True)
lor = col11_3.checkbox('LOR', value=True)

if st.button('Lancer'):
    train_mv = []
    test_mv = []

    if bmi:
        train_mv.append('BMI')
    else:
        test_mv.append('BMI')
    if chi:
        train_mv.append('CHI')
    else:
        test_mv.append('CHI')
    if cra:
        train_mv.append('CRA')
    else:
        test_mv.append('CRA')
    if dep:
        train_mv.append('DEP')
    else:
        test_mv.append('DEP')
    if fne:
        train_mv.append('FNE')
    else:
        test_mv.append('FNE')
    if gla:
        train_mv.append('GLA')
    else:
        test_mv.append('GLA')
    if lor:
        train_mv.append('LOR')
    else:
        test_mv.append('LOR')

    if len(train_mv) > 0 and len(test_mv) > 0:
        st.text('Données entrainement : {}'.format(train_mv))
        st.text('Données test : {}'.format(test_mv))

        container = st.beta_container()

        # Lecture du dataset
        txt_data = read_movies_data(params, train_mv, test_mv)

        params['LR']['FUZZ'] = lr_fuzz
        params['LR']['DEFUZZ'] = lr_defuzz
        params['LR']['CONV3'] = lr_conv3
        params['LR']['CONV4'] = lr_conv4
        params['LR']['CONV5'] = lr_conv5
        params['LR']['FC'] = lr_fc
        params['SIGMA'] = round(sigma, 2)
        params['DR'] = round(dropout_prob, 2)
        params['WD'] = round(8.86e-05, 7)

        ex_cnt += 1

        if params['SAVE_DIR_NEW'] is None:
            params['SAVE_DIR_NEW'] = params['SAVE_DIR']

        log_fname = os.path.join(params['SAVE_DIR_NEW'],
                                 'It{}_'.format(ex_cnt) + datetime.datetime.now().strftime('%H%M'))
        logger, log_fh = util.setup_logger('logger', log_fname + '.log')

        logger.info('Model parameters: {}'.format(params))
        logger.info('Learning rate: {}'.format(params['LR']))
        container.text('Learning rate: {}'.format(params['LR']))

        data = {'sub_train': txt_data['X_train'], 'sub_test': txt_data['X_test'], 'y_train': txt_data['y_train'],
                'y_test': txt_data['y_test']}
        data["vocab"] = sorted(list(set([w for sent in data["sub_train"] + data["sub_test"] for w in sent])))
        data["classes"] = sorted(list(set(data["y_train"])))
        data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
        data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

        params["MAX_SENT_LEN"] = max([len(sent) for sent in data["sub_train"] + data["sub_test"]])
        params["VOCAB_SIZE"] = len(data["vocab"])
        params["CLASS_SIZE"] = len(data["classes"])

        container.text("=" * 30 + "INFORMATION" + "=" * 30)
        container.text("DATASET: {}".format(params["DATASET"]))
        container.text("VOCAB_SIZE: {}".format(params["VOCAB_SIZE"]))
        container.text("MAX_SENT_LEN: {}".format(params["MAX_SENT_LEN"]))
        container.text("EPOCH: {}".format(params["EPOCH"]))

        container.text("=" * 30 + " FORMATION " + "=" * 30)
        util.log_params(logger, params)

        cnfn = model.CNFN(params).cpu()

        pars = {}
        for name, param in cnfn.named_parameters():
            if param.requires_grad:
                pars[name] = param

        p_conv3 = []
        for key, value in pars.items():
            if 'conv3' in key:
                p_conv3.append(value)
        p_conv4 = []
        for key, value in pars.items():
            if 'conv4' in key:
                p_conv4.append(value)
        p_conv5 = []
        for key, value in pars.items():
            if 'conv5' in key:
                p_conv5.append(value)
        p_fuzz = []
        for key, value in pars.items():
            if 'fuzz' in key:
                p_fuzz.append(value)
        p_defuz = []
        for key, value in pars.items():
            if 'defuz' in key:
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

        table_loss = []

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

            tr_loss = sum_loss / b_num
            table_loss.append(tr_loss)

            logger.info(
                "epoch:{: d}/ loss:{:6.4f}".format(e + 1, tr_loss))

            if e % params['N_SHOW'] == 0:
                container.write(" * epoch:{: d}/ loss:{:6.4f}".format(e + 1, tr_loss))

        dev_acc, dev_pred, dev_x, dev_y = model.test(data, cnfn, params)
        container.markdown('** --- Final test acc : {:.4f} --- **'.format(dev_acc))

        logger.removeHandler(log_fh)
        new_log_fname = log_fname + '_l{:.4f}_a{:.3f}.log'.format(tr_loss, dev_acc)
        os.rename(log_fname + '.log', new_log_fname)

        LR = params['LR']
        acc_list.append([ex_cnt, LR['FUZZ'], LR['DEFUZZ'], LR['CONV3'], LR['CONV4'], LR['CONV5'], LR['FC'],
                         params['SIGMA'], params['DR'], params['WD'],
                         tr_loss, dev_acc])

        acc_mat = np.array(acc_list)
        np.savetxt(os.path.join(params['SAVE_DIR_NEW'], "result.csv"), acc_mat, delimiter=',')
        avg_acc = np.average(acc_mat[:, -1])
        new_save_dname = '{}_it{}_aa{:.4f}'.format(params['SAVE_DIR'], ex_cnt, avg_acc)
        # os.rename(params['SAVE_DIR_NEW'], new_save_dname)
        # params['SAVE_DIR_NEW'] = new_save_dname
        # print('SAVE_DIR:', new_save_dname)

        # st.write(dev_acc)

        container.text("=" * 30 + "COURBE PERTE" + "=" * 30)
        chart_data = pd.DataFrame(np.array(table_loss), columns=['Perte'], index=['Epoch {}'.format(i+1) for i in range(len(table_loss))])
        container.line_chart(chart_data)

        container.text("=" * 30 + "PREDICTIONS" + "=" * 30)
        l = ['N', 'P']
        d = {'Texte': [' '.join(t) for t in dev_x], 'Label': [l[i] for i in dev_y], 'Prediction': [l[i] for i in dev_pred]}
        tab = pd.DataFrame(d)
        container.write(tab)
    else:
        if len(train_mv) == 0:
            st.warning('** Données entrainement vide **')
        if len(test_mv) == 0:
            st.warning('** Données test vide **')
else:
    st.markdown('Cliquez sur **Lancer** pour commencer l\'entrainement')

