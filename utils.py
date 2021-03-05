class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    # shrunk：降低
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def do_some_viz(preds, labels, des, name):
    """
    假设这里有两个输入， predictions和labels， 做两件事：
    1：保存图像  2：作图，当labels中元素增大时候，error的变化，
    :param preds: input predictions
    :param labels: input labels
    :return: xx
    """
    error = preds - labels
    ind = np.argsort(labels.ravel())
    resort = error[ind]
    plt.plot(labels[ind], np.square(np.abs(resort)), 'b.', linewidth=3, label="error subject to label")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.savefig(os.path.join(des, name))
    plt.show()

def copy_for_console():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_squared_error
    preds = np.load(r"C:\Users\wt\Desktop\outputviz\v_2\preds.npy")
    labels = np.load(r"C:\Users\wt\Desktop\outputviz\v_2\labels.npy")
    print(np.sqrt(mean_squared_error(preds, labels)))
    plt.plot(preds[-670:])
    plt.plot(labels[-670:])
    # plt.savefig(r"C:\Users\wt\Desktop\outputviz\v_2\0.756.png")
    plt.show()

def abc():
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.load(r"C:\Users\wt\Desktop\withoutpre\type1_30.npy", allow_pickle=True)
    b = np.load(r"C:\Users\wt\Desktop\withoutpre\type2_30.npy", allow_pickle=True)
    c = np.load(r"C:\Users\wt\Desktop\withoutpre\type3_30.npy", allow_pickle=True)
    a = a.tolist()
    b = b.tolist()
    c = c.tolist()
    plt.plot(a['val'], label='1')
    plt.plot(b['val'], label='2')
    plt.plot(c['val'], label='3')
    plt.legend(loc='upper right')