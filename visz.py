import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file1 = r"C:\Users\wt\Desktop\24_4_1.0_1.npy"
file2 = r"C:\Users\wt\Desktop\24_4_0.9_1.npy"
file3 = r"C:\Users\wt\Desktop\24_4_0.8_1.npy"
file4 = r"C:\Users\wt\Desktop\24_4_0.7_1.npy"
file5 = r"C:\Users\wt\Desktop\24_4_0.5_1.npy"

file6 = r"C:\Users\wt\Desktop\12_4_1.0_1.npy"
file7 = r"C:\Users\wt\Desktop\12_4_0.9_1.npy"
file8 = r"C:\Users\wt\Desktop\12_4_0.8_1.npy"
file9 = r"C:\Users\wt\Desktop\12_4_0.7_1.npy"
file10 = r"C:\Users\wt\Desktop\12_4_0.5_1.npy"

tr_1 = np.load(file6, allow_pickle=True)
tr_1 = tr_1.tolist()
tr_2 = np.load(file7, allow_pickle=True)
tr_2 = tr_2.tolist()
tr_3 = np.load(file8, allow_pickle=True)
tr_3 = tr_3.tolist()
tr_4 = np.load(file9, allow_pickle=True)
tr_4 = tr_4.tolist()
tr_5 = np.load(file10, allow_pickle=True)
tr_5 = tr_5.tolist()

print(min(tr_1['val']))
print(min(tr_2['val']))
print(min(tr_3['val']))
print(min(tr_4['val']))
print(min(tr_5['val']))

font3 = {'family' : 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }
font1 = {'family' : 'Times New Roman',
        'weight': 'normal',
        'size': 28,
        }
font2 = {'family' : 'Times New Roman',
        'weight': 'normal',
        'size': 36,
        }

plt.figure(figsize=(7, 5))
plt.plot(tr_1['train'][:24], 'b-', label="Rate=1.0", marker='x')
plt.plot(tr_2['train'], label="Rate=0.9", marker='+')
plt.plot(tr_3['train'], label="Rate=0.8", marker='o')
plt.plot(tr_4['train'], label="Rate=0.7", marker='x')
plt.plot(tr_5['train'], "b--", label="Rate=0.5", marker='>')
plt.legend(loc="upper right")
plt.xlabel("Epoch", font3)
plt.ylabel("Reconstruction Loss", font3)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.title("Train Reconstruction Loss(embed=12)", fontproperties='Times New Roman', weight='bold', size=15)
plt.tight_layout()
plt.savefig(r"C:\Users\wt\Desktop\b22.pdf", dpi=800)
plt.show()
# plt.plot(tr_1['val'][:24], 'b.', label="Rate=1.0")
# plt.plot(tr_2['val'], label="Rate=0.9")
# plt.plot(tr_3['val'], label="Rate=0.8")
# plt.plot(tr_4['val'], label="Rate=0.7")
# plt.plot(tr_5['val'], "b--", label="Rate=0.5")
#
# plt.legend(loc="upper right")
# plt.xlabel("epoch")
# plt.ylabel("reconstruction loss")
# plt.title("Validation reconstruction loss(embed=12)")
# plt.show()
# #
# plt.plot(tr_1['train'][:24], "b.", label="1.0")
# plt.plot(tr_2['train'], label="0.9")
# plt.plot(tr_3['train'], label="0.8")
# plt.plot(tr_4['train'], label="0.7")
# plt.plot(tr_5['train'], "b--", label="0.5")
# plt.legend(loc="upper right")
# plt.xlabel("epoch")
# plt.ylabel("reconstruction loss")
# plt.title("Train reconstruction loss(embed=12)")
# plt.show()

# plt.plot(tr_1['batch'][:24], "b.", label="tr_1")
# plt.plot(tr_2['batch'], label="tr_2")
# plt.plot(tr_3['batch'], label="tr_3")
# plt.plot(tr_4['batch'], label="tr_4")
# plt.plot(tr_5['batch'], "b--", label="tr_5")
# plt.legend(loc="upper right")
# plt.xlabel("epoch")
# plt.ylabel("reconstruction loss")
# plt.title("Batch reconstruction loss(embed=12)")
# plt.show()