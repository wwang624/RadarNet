import numpy as np
import math


def compute_iou(pred, gt):
    gt[gt >= 0.3] = 1
    gt[gt < 0.3] = 0
    pred[pred >= 0.3] = 1
    pred[pred < 0.3] = 0
    intersection = np.sum(np.logical_and(gt, pred))
    union = np.sum(np.logical_or(gt, pred)) + 1e-7
    iou = (intersection+1e-7) / union

    return iou


def precision_recall(pred, gt):
    gt[gt >= 0.3] = 1
    gt[gt < 0.3] = 0
    pred[pred >= 0.3] = 1
    pred[pred < 0.3] = 0
    intersection = np.sum(np.logical_and(gt, pred))
    precision = (intersection + 1e-7) / (np.sum(pred) + 1e-7)
    recall = (intersection + 1e-7) / (np.sum(gt) + 1e-7)

    return precision, recall


def f1_score(precision, recall):
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return f1


def radar_score(precision, recall):
    # score = np.exp(-1.5 * (2 - precision - recall))
    score = np.log(math.e - 2 + precision + recall)

    return score
