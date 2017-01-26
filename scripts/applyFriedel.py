from PyQt4 import QtGui
import pyqtgraph as pg
import os
import numpy as np
import psana

def setup(experimentName,runNumber,detInfo):
    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)
    return run, times, det, evt

class FriedelSym(object):
    def __init__(self, dim, centre):
        self.dim = dim
        self.centre = centre
        # Centre by zeropadding
        pdim = np.zeros_like(centre)  # padded dimension
        for i, val in enumerate(centre):
            pdim[i] = 2 * max(val, dim[i] - val + 1) + 1
        shift = np.floor(pdim / 2.) + 1 - centre
        endGap = pdim - dim
        self.pad = []
        for i in zip(shift, endGap - 1):
            self.pad.append(i)

    def __zeropad(self, img):
        zeropad = np.lib.pad(img, (self.pad), 'constant')
        return zeropad

    def applyFriedel(self, img, mask=None, mode='same'):
        zimg = self.__zeropad(img)
        if mask is not None:
            zmask = self.__zeropad(mask)
            zimg[np.where(zmask == 0)] = 0
        imgSym = zimg.ravel() + zimg.ravel()[::-1]
        imgSym.shape = zimg.shape
        if mask is None:
            imgSym /= 2.
        else:
            maskSym = zmask.ravel() + zmask.ravel()[::-1]
            maskSym.shape = zimg.shape
            a = np.zeros_like(imgSym)
            a[np.where(maskSym > 0)] = imgSym[np.where(maskSym > 0)] / maskSym[np.where(maskSym > 0)]
            imgSym = a
        if mode == 'same':
            slices = [slice(a, imgSym.shape[i] - b) for i, (a, b) in enumerate(self.pad)]
            cropImg = imgSym[slices]
            return cropImg
        elif mode == 'full':
            return imgSym

experimentName = 'amo86615'
run = 197
detInfo = 'pnccdBack'
eventInd = 264
run, times, det, evt = setup(experimentName, run, detInfo)
evt = run.event(times[eventInd])
data = det.image(evt)

centre = det.point_indexes(evt, pxy_um=(0, 0))
fs = FriedelSym(data.shape, centre)
dataSym = fs.applyFriedel(data, mask=None, mode='same')

pg.image(data, title="LCLS images",levels=(0,200))
pg.image(dataSym, title="LCLS sym images",levels=(0,200))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
