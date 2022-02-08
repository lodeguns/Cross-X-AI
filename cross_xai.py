import numpy as np
import os
import csv
from matplotlib import pyplot as plt
import scipy
from scipy import stats
import tensorflow as tf



def binaryMaskIOU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


def ssi_acc(y_true, y_pred):
    return  tf.reduce_mean(tf.image.ssim(tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1), max_val=1., filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03))

depth_test_path = os.path.join("test_img_recon", "X_test_depth")
mask_test_path = os.path.join("test_img_recon", "X_test")



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def f1_loss(y_true, y_pred, beta=1):
    '''Calculate F1 score.

    The original implmentation is written by Michal Haltuf on Kaggle.

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''


    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + epsilon)

    return f1


def PSNR(original, compressed):
    import math
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

bad_masks = []
rows_for_csv = []
count_diff_good = 0

for id_p in range(750):
    res = np.load(os.path.join(depth_test_path, str(id_p), "result.npy"))
    gmap_depth = res[0]
    gmappp_depth = res[1]
    gmap_unet = np.load(os.path.join(mask_test_path, str(id_p), "heatmap.npy"))
    gmappp_unet = np.load(os.path.join(mask_test_path, str(id_p), "heatmappp.npy"))
    depth = res[8]
    gt_mask = res[9]

    pred_mask = np.load(os.path.join(mask_test_path, str(id_p), "pred.npy"))
    input_img = np.load(os.path.join(mask_test_path, str(id_p), "input.npy"))
    if np.isnan(np.sum(gt_mask)) or np.isnan(np.sum(depth)) or np.isnan(np.sum(pred_mask)):
        bad_masks.append(id_p)
    else:
        gmap_depth = (gmap_depth - np.min(gmap_depth)) / np.ptp(gmap_depth)
        gmappp_depth = (gmappp_depth - np.min(gmappp_depth)) / np.ptp(gmappp_depth)
        gmap_unet = (gmap_unet - np.min(gmap_unet)) / np.ptp(gmap_unet)
        gmappp_unet = (gmappp_unet - np.min(gmappp_unet)) / np.ptp(gmappp_unet)
        ic_depth = np.mean(np.abs(gmap_depth - gmappp_depth))
        ic_unet = np.mean(np.abs(gmap_unet - gmappp_unet))

        if np.isnan(np.sum(gmappp_depth)):
            bad_masks.append(id_p)
        else:
            import math

            depth = (depth - np.min(depth)) / np.ptp(depth)
            pred_mask[pred_mask >= 0.5] = 1
            pred_mask[pred_mask < 0.5] = 0

            gt_mask[gt_mask >= 0.5] = 1
            gt_mask[gt_mask < 0.5] = 0

            masked_gmappp_gt_unet = np.asarray(np.multiply(gmappp_unet,gt_mask), dtype=np.float32)
            masked_gmappp_pred_unet = np.asarray(np.multiply(gmappp_unet, pred_mask), dtype=np.float32)
            masked_gmappp_gt_depth = np.asarray(np.multiply(gmappp_depth, gt_mask), dtype=np.float32)
            masked_gmappp_pred_depth = np.asarray(np.multiply(gmappp_depth, pred_mask), dtype=np.float32)
            plt.imshow(gmappp_unet)
            plt.savefig(os.path.join(depth_test_path, str(id_p), "gmappp_unet.png"))
            plt.imshow(gmappp_depth)
            plt.savefig(os.path.join(depth_test_path, str(id_p), "gmappp_depth.png"))

            plt.imshow(masked_gmappp_gt_unet)
            plt.savefig(os.path.join(depth_test_path, str(id_p), "depth_mask_gt_unet.png"))
            plt.clf()
            plt.imshow(masked_gmappp_pred_unet)
            plt.savefig(os.path.join(depth_test_path, str(id_p), "depth_mask_pred_unet.png"))
            plt.clf()
            plt.imshow(masked_gmappp_gt_depth)
            plt.savefig(os.path.join(depth_test_path, str(id_p), "depth_mask_gt_depth.png"))
            plt.clf()
            plt.imshow(masked_gmappp_pred_depth)
            plt.savefig(os.path.join(depth_test_path, str(id_p), "depth_mask_pred_depth.png"))
            plt.clf()
            iou = binaryMaskIOU(pred_mask, gt_mask)
            f1 = f1_loss(pred_mask, gt_mask)
            ssim = (1 + ssi_acc(tf.convert_to_tensor(depth, dtype=tf.float32), tf.convert_to_tensor(rgb2gray(input_img/ 255.) , dtype=tf.float32))) / 2
            mse = ((rgb2gray(input_img/ 255.) - depth)**2).mean()
            rmse = math.sqrt(mse)
            psnr = PSNR((rgb2gray(input_img/ 255.)), depth)
            # GradCam and GradCam++ pixel mean inside the GT mask
            in_gmap_depth_gt = np.sum(gmap_depth[gt_mask == 1]) / len(gmap_depth[gt_mask == 1])
            in_gmappp_depth_gt = np.sum(gmappp_depth[gt_mask == 1]) / len(gmappp_depth[gt_mask == 1])
            in_gmap_unet_gt = np.sum(gmap_unet[gt_mask == 1]) / len(gmap_unet[gt_mask == 1])
            in_gmappp_unet_gt = np.sum(gmappp_unet[gt_mask == 1]) / len(gmappp_unet[gt_mask == 1])
            # GradCAm and GradCam++ pixel mean inside the UNET predicted mask
            in_gmap_depth_pred = np.sum(gmap_depth[pred_mask == 1]) / len(gmap_depth[pred_mask == 1])
            in_gmappp_depth_pred= np.sum(gmappp_depth[pred_mask == 1]) / len(gmappp_depth[pred_mask == 1])
            in_gmap_unet_pred = np.sum(gmap_unet[pred_mask == 1]) / len(gmap_unet[pred_mask == 1])
            in_gmappp_unet_pred = np.sum(gmappp_unet[pred_mask == 1]) / len(gmappp_unet[pred_mask == 1])

            # The MSE between GradCam and GradCam++ inside the predicted and the GT mask
            in_gmappp_mse_pred = ((gmappp_depth[pred_mask == 1] - gmappp_unet[pred_mask == 1])**2).mean()
            in_gmappp_mse_gt = ((gmappp_depth[gt_mask == 1] - gmappp_unet[gt_mask == 1])**2).mean()
            in_gmap_mse_pred = ((gmap_depth[pred_mask == 1] - gmap_unet[pred_mask == 1])**2).mean()
            in_gmap_mse_gt = ((gmap_depth[gt_mask == 1] - gmap_unet[gt_mask == 1])**2).mean()

            # Depth map pixels mean inside and outside the predicted and the GT mask
            out_m_gt   = np.sum(depth[gt_mask==0])/len(depth[gt_mask==0])
            in_m_gt    = np.sum(depth[gt_mask==1])/len(depth[gt_mask==1])
            out_m_pred = np.sum(depth[pred_mask==0])/len(depth[pred_mask==0])
            in_m_pred  = np.sum(depth[pred_mask==1])/len(depth[pred_mask==1])

            rows_for_csv.append([id_p,  out_m_gt,in_m_gt,out_m_pred,in_m_pred, np.abs(out_m_gt-in_m_gt), np.abs(out_m_pred-in_m_pred),
                                np.abs(in_gmap_depth_gt-in_gmap_unet_gt),  np.abs(in_gmappp_depth_gt-in_gmappp_unet_gt),
                                np.abs(in_gmap_depth_pred-in_gmap_unet_pred),  np.abs(in_gmappp_depth_pred-in_gmappp_unet_pred),
                                 in_gmappp_mse_pred, in_gmappp_mse_gt, ic_depth, ic_unet, iou, f1, mse, ssim.numpy(), rmse, psnr, in_gmap_mse_pred, in_gmap_mse_gt])



header = ['id_p', 'out_m_gt', 'in_m_gt', 'out_m_pred', 'in_m_pred', 'diff_gt', 'diff_pred',
          'diff_gmap_gt', 'diff_gmappp_gt', 'diff_gmap_pred', 'diff_gmappp_pred', 'mse_gmappp_pred', 'mse_gmappp_gt', 'ic_depth',
          'ic_unet', 'iou', 'f1', 'mse', 'ssim', 'rmse', 'psnr', 'mse_gmappp_pred', 'mse_gmappp_gt']




with open(os.path.join('test_img_recon', 'diffs.csv'), 'w+', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data

    for r in rows_for_csv:
        writer.writerow(r)

f.close()


rows_numpy = np.asarray(rows_for_csv)
print(rows_numpy.shape)




col_gt = rows_numpy[:, 11]
(mu_gt, sigma_gt) = stats.norm.fit(col_gt)
n_gt, bins_gt, patches_gt = plt.hist(col_gt, 30, facecolor='green')
#y_gt = stats.norm.pdf( bins_gt, mu_gt, sigma_gt)
#l_gt = plt.plot(bins_gt, y_gt, 'r--', linewidth=2)
plt.grid(True)
plt.savefig("./histogram_mse_PRED.png")
plt.clf()
col_pred = rows_numpy[:, 12]
(mu_pred, sigma_pred) = stats.norm.fit(col_pred)
n_pred, bins_pred, patches_pred = plt.hist(col_pred, 30, facecolor='green')
#y_pred = stats.norm.pdf( bins_pred, mu_pred, sigma_pred)
#l_pred = plt.plot(bins_pred, y_pred, 'r--', linewidth=2)
plt.grid(True)
plt.savefig("./histogram_mse_GT.png")


col_0 = 1 - rows_numpy[:, 11]
col_1 = 1 - rows_numpy[:, 12]

print(np.corrcoef(col_0, col_1))