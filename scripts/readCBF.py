import cbf
from PyQt4 import QtGui
import pyqtgraph as pg
import os
import numpy as np

startInd = 987
endInd = 1007
numData = endInd - startInd + 1
path = '/reg/d/psdm/cxi/cxitut13/scratch/yoon82/wtich_274k_10/cbf'
data = None
for i, val in enumerate(range(startInd,endInd)):
    fname = os.path.join(path,'wtich_274_10_1_'+str(val).zfill(5)+'.cbf')
    content = cbf.read(fname)
    if data is None:
        data = np.zeros((numData,content.data.shape[0],content.data.shape[1]))
    data[i,:,:] = content.data   
    with open(fname, 'r') as f:
        for i in f.readlines():
            if "Start_angle" in i: print i

pg.image(data, title="crystal rotation series",levels=(0,50))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
