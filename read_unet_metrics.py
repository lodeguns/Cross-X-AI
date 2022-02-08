from matplotlib import pyplot as plt
import json
import numpy as np
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

f = open('metrics_unet_training.json')

data = json.load(f)
print(data)

train_iou = smooth_curve(np.asarray(data["iou_score"]))
train_f1 = smooth_curve(np.asarray(data["f1-score"]))

val_iou = smooth_curve(np.asarray(data["val_iou_score"]))
val_f1 = smooth_curve(np.asarray(data["val_f1-score"]))


plt.plot(train_iou, label="training IoU")
plt.plot(train_f1, label="training f1-score")
plt.title("Training segmentation scores best ep: %d IoU: %f" %(np.argmax(train_iou) + 1, np.max(train_iou)))
plt.legend()
plt.show()

plt.plot(val_iou, label="validation IoU")
plt.plot(val_f1, label="validation f1-score")
plt.title("Validation segmentation scores best ep: %d IoU: %f" %(np.argmax(val_iou) + 1, np.max(val_iou)))
plt.legend()
plt.show()





