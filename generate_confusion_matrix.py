import matplotlib
matplotlib.use('pgf')
import seaborn as sn
import pandas as pd

import matplotlib.pyplot as plt

array = [[ 7 , 5 , 1 , 3 , 0 , 0, 0, 1],
 [ 2, 32, 0, 4, 0, 0, 1, 0],
 [ 2, 0, 11, 0, 2, 6, 0, 10],
 [ 6, 8, 5, 13, 0, 2 , 5 , 3],
 [ 0, 2, 0, 0, 31, 0, 4, 4],
 [ 2, 2, 5, 4, 1, 19, 0, 5],
 [ 1, 2, 0, 2, 6, 1, 26, 7],
 [ 1, 1, 3, 0, 1, 9, 2, 18]]
df_cm = pd.DataFrame(array, range(8),
                  range(8))
# plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
plt.savefig('confusion_matrix.pdf')
plt.savefig('confusion_matrix.pgf') # for latex