import os
import tqdm
import sys

import pandas as pd
import numpy as np

from PIL import Image

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from eval_functions import *
from misc import *

BETA = 1.0

def evaluate(name):
    result_path = "results"
    pred_root = r'/data1/shoupeiyao/workspace/UCOD/last/outs/' + name + '/'
    gt_root = r'/data1/shoupeiyao/Data/dirnetdata/datas/TestDataset/' + name + '/GT/'
    if os.path.isdir(result_path) is False:
        os.makedirs(result_path)
    results = []

    pred_root = os.path.join(pred_root)
    gt_root = os.path.join(gt_root)

    preds = os.listdir(pred_root)
    gts = os.listdir(gt_root)
    preds = sort(preds)
    gts = sort(gts)
        
    preds = [i for i in preds]
    gts = [i for i in gts]

    len_ = len(preds)

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MAE = Mae()
    MSE = Mse()
    MBA = BoundaryAccuracy()
    IOU = IoU()
    BIOU = BIoU()
    TIOU = TIoU()

    samples = enumerate(zip(preds, gts))
    for i, sample in samples:
        pred, gt = sample

        pred_mask = np.array(Image.open(os.path.join(pred_root, pred)).convert('L'))
        gt_mask = np.array(Image.open(os.path.join(gt_root, gt)).convert('L'))
        
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)).convert('L').resize((gt_mask.shape[1],gt_mask.shape[0])))

        FM.step( pred=pred_mask, gt=gt_mask)
        WFM.step(pred=pred_mask, gt=gt_mask)
        SM.step( pred=pred_mask, gt=gt_mask)
        EM.step( pred=pred_mask, gt=gt_mask)
        MAE.step(pred=pred_mask, gt=gt_mask)
        MSE.step(pred=pred_mask, gt=gt_mask)
        MBA.step(pred=pred_mask, gt=gt_mask)
        IOU.step(pred=pred_mask, gt=gt_mask)
        BIOU.step(pred=pred_mask, gt=gt_mask)
        TIOU.step(pred=pred_mask, gt=gt_mask)
            
    result = []
    Sm =  SM.get_results()["sm"]
    wFm = WFM.get_results()["wfm"]
    mae = MAE.get_results()["mae"]
    mse = MSE.get_results()["mse"]
    mBA = MBA.get_results()["mba"]
    Fm =  FM.get_results()["fm"]
    Em =  EM.get_results()["em"]
    Iou = IOU.get_results()["iou"]
    BIou = BIOU.get_results()["biou"]
    TIou = TIOU.get_results()["tiou"]
    adpEm = Em["adp"]
    avgEm = Em["curve"].mean()
    maxEm = Em["curve"].max()
    adpFm = Fm["adp"]
    avgFm = Fm["curve"].mean()
    maxFm = Fm["curve"].max()
    avgIou = Iou["curve"].mean()
    maxIou = Iou["curve"].max()
    avgBIou = BIou["curve"].mean()
    maxBIou = BIou["curve"].max()
    avgTIou = TIou["curve"].mean()
    maxTIou = TIou["curve"].max()
    print('Sm:', Sm, 'mae:', mae, 'maxFm:', maxFm, 'avgFm:', avgFm, 'adpFm:', adpFm, 'maxEm:', maxEm, 'avgEm:', avgEm, 'adpEm:', adpEm)


    
if __name__ == "__main__":
    evaluate()
