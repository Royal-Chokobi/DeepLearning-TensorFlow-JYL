import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)

print(hypothesis)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


x = np.array([1.0, 2.0, 3.0])

y = softmax(x)

print(y)


def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)
xy = min_max_scaler(xy)
print(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# plt.figure(figsize=(4,6)) # 4 incheswide 6 inches tall

plt.plot(x_data, y_data, "ro")
plt.plot([0, 1])
# plt.axis([0, 10, 0, 10]) # [xmin, xmax, ymin, ymax]
plt.annotate('square it', (np.mean(x_data), np.mean(x_data)))
plt.show()