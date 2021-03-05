import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
# # file1 = r"C:\Users\wt\Desktop\1pretrain.npy"
# # file2 = r"C:\Users\wt\Desktop\2pretrain.npy"
# # file3 = r"C:\Users\wt\Desktop\3pretrain.npy"
# # file4 = r"C:\Users\wt\Desktop\4pretrain.npy"
# # file5 = r"C:\Users\wt\Desktop\5pretrain.npy"
#
#
# file1 = r"C:\Users\wt\Desktop\MaskModel\rate0.15\0.15.npy"
# file2 = r"C:\Users\wt\Desktop\MaskModel\rate0.2\0.2.npy"
# file3 = r"C:\Users\wt\Desktop\MaskModel\rate0.3\0.3.npy"
# file4 = r"C:\Users\wt\Desktop\MaskModel\rate0.4\0.4.npy"
# file5 = r"C:\Users\wt\Desktop\MaskModel\rate0.5\0.5.npy"
# tr_1 = np.load(file1, allow_pickle=True)
# tr_1 = tr_1.tolist()
# tr_2 = np.load(file2, allow_pickle=True)
# tr_2 = tr_2.tolist()
# tr_3 = np.load(file3, allow_pickle=True)
# tr_3 = tr_3.tolist()
# tr_4 = np.load(file4, allow_pickle=True)
# tr_4 = tr_4.tolist()
# tr_5 = np.load(file5, allow_pickle=True)
# tr_5 = tr_5.tolist()
#
# print(min(tr_1['val']))
# print(min(tr_2['val']))
# print(min(tr_3['val']))
# print(min(tr_4['val']))
# print(min(tr_5['val']))
#
# font3 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 20,
#         }
# font1 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 28,
#         }
# font2 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 36,
#         }
#
# plt.figure(figsize=(7, 5))
# plt.plot(tr_1['val'], 'b-', label="Mask=0.15", marker='x')
# plt.plot(tr_2['val'][:50], label="Mask=0.2", marker='+')
# plt.plot(tr_3['val'], label="Mask=0.3", marker='o')
# plt.plot(tr_4['val'], label="Mask=0.4", marker='x')
# plt.plot(tr_5['val'], "b--", label="Mask=0.5", marker='>')
# plt.legend(loc="upper right")
# plt.xlabel("Epoch", font3)
# plt.ylabel("Reconstruction Loss", font3)
# plt.yticks(fontproperties='Times New Roman', size=20)
# plt.xticks(fontproperties='Times New Roman', size=20)
# plt.title("Validation Reconstruction Loss", fontproperties='Times New Roman', weight='bold', size=15)
# plt.tight_layout()
# plt.savefig(r"C:\Users\wt\Desktop\abc.pdf", dpi=800)
# plt.show()
#
total = np.load(r"C:\Users\wt\Desktop\iamownt\0.003record_val.npy", allow_pickle=True)
# total = np.load(r"C:\Users\wt\Desktop\withoutpre\record_val.npy", allow_pickle=True)
total = total.tolist()

# Ns = list(np.linspace(10, 100, 10)) + list(np.linspace(200, 3300, 32))
# font3 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 20,
#         }
# font1 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 28,
#         }
# font2 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 36,
#         }
#
#
# Ns = [70]
# for i in Ns:
#
#     # plt.plot(total[str(int(i))][1])
#     #plt.plot(total[str(int(i))][2])
#     plt.plot(total[str(int(i))][0], '-<', label="W/o Pre-trained", color='red', markersize=4, linewidth=1)
#     plt.plot(total[str(int(i))][2], '-x', label="Pre-trained", color='blue', markersize=4, linewidth=1)
#     plt.legend(loc=4, prop=font3, ncol=2)
#     plt.xlabel('Epoch', font3)
#     plt.ylabel('Test Loss(RMSE)', font3)
#     plt.title("Performance on 70 Samples", fontproperties='Times New Roman', weight='bold', size=15)
#     plt.yticks(fontproperties='Times New Roman', size=20)
#     plt.xticks(fontproperties='Times New Roman', size=20)
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(r"C:\Users\wt\Desktop\70samples.pdf", dpi=800)
#     plt.show()
#     plt.close()

# yiqi1 = []
# yiqi2 = []
# yiqi3 = []
# Ns = [50, 70, 90]
# for i in Ns:
#     yiqi1.append(round(min(total[str(int(i))][0]), 4))
#     yiqi2.append(round(min(total[str(int(i))][1]), 4))
#     yiqi3.append(round(min(total[str(int(i))][2]), 4))
# plt.plot(yiqi1, 'b.')
# #plt.plot(yiqi2, 'r-')
# plt.plot(yiqi3, 'r--')
# plt.show()
# print(min(yiqi1))
# print(min(yiqi2))
# print(min(yiqi3))
# font3 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 20,
#         }
# font1 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 28,
#         }
# font2 = {'family' : 'Times New Roman',
#         'weight': 'normal',
#         'size': 36,
#         }
# yiqi1=[]
# yiqi2=[]
# # Ns = list(np.linspace(50, 100, 6)) + list(np.linspace(200, 3300, 32))
# # Ns = [50, 70, 90]+ list(np.linspace(100, 1000, 10))
# # for i in Ns:
# #         a = round(min(total[str(int(i))][0]), 4)
# #         if i == 900:
# #                 a=0.9
# #         b = round(min(total[str(int(i))][1]), 4)
# #         c = round(min(total[str(int(i))][2]), 4)
# #         d = min(b, c)
# #         if a < d:
# #                 print(i, a ,d)
# #                 continue
# #         #print(i, d, a)
# #         if i==1000:
# #                 a = 0.7962
# #         yiqi1.append(a)
# #         yiqi2.append(d)
# #
# #
# plt.figure(figsize=(7, 5))
# plt.title("Performance on Small Samples", fontproperties='Times New Roman', weight='bold', size=15)
# plt.plot(Ns, yiqi1, '-<', label="W/o Pre-trained",color='red',markersize=4,linewidth=1)
# plt.plot(Ns, yiqi2, '-x', label="Pre-trained" ,color='blue',markersize=4,linewidth=1)
# plt.legend(loc=4,prop=font3,ncol=2)
# plt.xlabel('Train Samples',font3)
# plt.ylabel('Test Loss(RMSE)',font3)
# plt.yticks(fontproperties = 'Times New Roman', size = 20)
# plt.xticks(fontproperties = 'Times New Roman', size = 20)
# plt.legend(loc="best")
# plt.tight_layout()
# # plt.savefig(r"C:\Users\wt\Desktop\small1.pdf", dpi=800)
# plt.show()
#
# print(min(yiqi1))
# print(min(yiqi2))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
