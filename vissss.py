import matplotlib.pyplot as plt
import numpy as np

def rmseloss(pre,real):
    mse = 0
    mae = 0
    for i in range(len(pre)):
        mse += (0.01*pre[i]-0.01*real[i])**2
        mae += abs(0.01*pre[i]-0.01*real[i])
    mse = mse/len(real)
    mae = mae/len(real)
    print(np.sqrt(mse)*100,mae*100)
    return mse
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
#VAE = np.load('pre_VAE_129.npy')
VAE = np.load(r"C:\Users\wt\Desktop\experimente_features\0.5666.npy")
testlabel = np.load(r'C:\Users\wt\Desktop\outputviz\v_2\1020\labels.npy')
#plt.figure(figsize=(20, 7))
plt.figure(figsize=(7, 5))
#ax = fig.add_subplot(111)
#plt.figure(figsize=(10, 10))
#ax.set(title="Compare", xlabel="Y values (actual)", ylabel="Y values (predicted)")
plotx = np.arange(len(testlabel[65:]))
plt.plot(plotx[:],testlabel[65:],'-o',label = 'Label',color='red',markersize=4,linewidth=1)
plt.plot(plotx[:],VAE[65:],'-x',label = 'Pre',color='blue',markersize=4,linewidth=1)
rmseloss(VAE[65:],testlabel[65:])
#plt.scatter(plotx[65:], svm[65:],alpha=1,c='blue',marker='x',s=20)
#plt.scatter(testrealarr, sselm,alpha=1,label = 'SS-ELM',c='blue',marker='x')
#plt.scatter(testrealarr, deepfm,alpha=1,label = 'SS-PdeepFM',c='r',marker='+')
#plt.scatter(testlabel, sigma1,alpha=0.5,label = 'σ=0.3',c='blue')
#plt.scatter(testlabel, sigma5,alpha=0.5,label = 'σ=0.1',c='green')
plt.ylim([0,6])
#plt.xlim([0,6])
#plt.plot([0,6], [0,6], color='black',label='Real Label')
#plt.legend(prop=font3)
plt.legend(loc=4,prop=font3,ncol=2)
plt.xlabel('Test Samples',font1)
plt.ylabel('Impurity Content(%)',font1)
plt.yticks(fontproperties = 'Times New Roman', size = 25)
plt.xticks(fontproperties = 'Times New Roman', size = 25)
plt.tight_layout()
plt.savefig(r'C:\Users\wt\Desktop\aaaa.pdf',dpi=800)
plt.show()