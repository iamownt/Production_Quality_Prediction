import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file1 = r"C:\Users\wt\Desktop\MaskModel\rate0.15\0.15.npy"
file2 = r"C:\Users\wt\Desktop\MaskModel\rate0.2\0.2.npy"
file3 = r"C:\Users\wt\Desktop\MaskModel\rate0.3\0.3.npy"
file4 = r"C:\Users\wt\Desktop\MaskModel\rate0.4\0.4.npy"
file5 = r"C:\Users\wt\Desktop\MaskModel\rate0.5\0.5.npy"

tr_1 = np.load(file1, allow_pickle=True)
tr_1 = tr_1.tolist()
tr_2 = np.load(file2, allow_pickle=True)
tr_2 = tr_2.tolist()
tr_3 = np.load(file3, allow_pickle=True)
tr_3 = tr_3.tolist()
tr_4 = np.load(file4, allow_pickle=True)
tr_4 = tr_4.tolist()
tr_5 = np.load(file5, allow_pickle=True)
tr_5 = tr_5.tolist()

plt.plot(tr_1['val'], 'b.', label="Mask=0.15")
plt.plot(tr_2['val'][:50], label="Mask=0.2")
plt.plot(tr_3['val'], label="Mask=0.3")
plt.plot(tr_4['val'], label="Mask=0.4")
plt.plot(tr_5['val'], "b--", label="Mask=0.5")
plt.legend(loc="upper right")
plt.xlabel("epoch")
plt.ylabel("reconstruction loss")
plt.title("Validation reconstruction loss")
plt.show()
