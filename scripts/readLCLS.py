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

experimentName = 'amo86615'
run = 197
detInfo = 'pnccdBack'
eventInd = 264
run, times, det, evt = setup(experimentName, run, detInfo)
evt = run.event(times[eventInd])
data = det.image(evt)

pg.image(data, title="LCLS images",levels=(0,200))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
