from PyQt4 import QtGui
import pyqtgraph as pg
import os, time
import numpy as np
import psana
import h5py
from numba import jit

def setup(experimentName,runNumber,detInfo):
    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)
    return run, times, det, evt

def convert2Photons(imgD,weightD,avgADU):
    photonsD = np.ceil(imgD*weightD/avgADU)
    # we only reserve 8 bits for photon counts in hdf5
    # so anything larger should become uint16
    maxThresh=65535
    maxInd = np.argwhere(photonsD>maxThresh)
    photonsD[maxInd[:,0],maxInd[:,1]] = maxThresh

    return photonsD.astype('uint16')

@jit
def downsample(img,threshold,mask,factor):
    if factor == 1:
        return img, np.ones_like(img)
    else:
        superpixelSize = np.float(factor**2)
        # Subtract threshold
        I = img - threshold
        # mask out bad pixels
        I *= mask
        # zero out pixels less than 0
        negInd = np.argwhere(I<=0)
        I[negInd[:,0],negInd[:,1]] = 0
        # zero out pixels that are inf
        infInd = np.argwhere(I==np.inf)
        I[infInd[:,0],infInd[:,1]] = 0
        #####
        # zeropad, only need to do once
        numRows,numCols = img.shape
        zeropadTop = 0
        zeropadLeft = 0
        if np.mod(img.shape[0],factor) > 0:  # zeropad in y
            numRows = np.int(np.ceil(img.shape[0]*1.0/factor)*factor)
            diffY = numRows-img.shape[0]
            zeropadTop = np.ceil(diffY/2.)
            print "numRows,diffY,zeropadTop: ",numRows,diffY,zeropadTop
        if np.mod(img.shape[1],factor) > 0:  # zeropad in x
            numCols = np.int(np.ceil(img.shape[1]*1.0/factor)*factor)
            diffX = numCols-img.shape[1]
            zeropadLeft = np.ceil(diffX/2.)
            print "numCols,diffX,zeropadLeft: ",numCols,diffX,zeropadLeft
        #####
        imgZ = np.zeros((numRows,numCols))
        goodpixZ = np.zeros((numRows,numCols))
        imgZ[zeropadTop:zeropadTop+img.shape[0],zeropadLeft:zeropadLeft+img.shape[1]] = I
        goodpixZ[zeropadTop:zeropadTop+img.shape[0],zeropadLeft:zeropadLeft+img.shape[1]] = I > 0
        # downsample
        downsampleRows = numRows/factor
        downsampleCols = numCols/factor
        imgD = np.zeros((downsampleRows,downsampleCols))
        goodpixD = np.zeros((downsampleRows,downsampleCols))
        weightD = np.zeros((downsampleRows,downsampleCols))
        for i in range(downsampleRows):  # superpixel
            for j in range(downsampleCols):
                imgD[i,j] = np.sum(imgZ[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
                goodpixD[i,j] = np.sum(goodpixZ[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
                # Find pixels in imgD with ADUs higher than threshD
                if goodpixD[i,j] > 0:
                    weightD[i,j] = superpixelSize/goodpixD[i,j] # minimum weight is 1
        return imgD,weightD

experimentName = 'cxitut13'
run = 10
detInfo = 'DscCsPad'
eventInd = 554
time6 = time.time()
run, times, det, evt = setup(experimentName, run, detInfo)
time4 = time.time()
print "setup: ", time4-time6
evt = run.event(times[eventInd])
time5 = time.time()
print "evt: ", time5-time4
img = det.image(evt)
time7 = time.time()
print "image: ", time7-time5

threshold = 0
mask_calibOn = True
mask_statusOn = True
mask_edgesOn = True
mask_centralOn = True
mask_unbondOn = True
mask_unbondnrsOn = True
psanaMask = det.mask(evt, calib=mask_calibOn, status=mask_statusOn, edges=mask_edgesOn, central=mask_centralOn, unbond=mask_unbondOn, unbondnbrs=mask_unbondnrsOn)
mask = det.image(evt, psanaMask)
factor = 2

time0 = time.time()

imgD, weightD = downsample(img, threshold, mask, factor)
downsampledImg = imgD*weightD

time1 = time.time()
print "imgD time: ", time1-time0

avgADU = 20
ph = det.photons(evt, adu_per_photon=avgADU)
photons = det.image(evt, ph)

time2 = time.time()
print "photons time: ", time2-time1

imgD, weightD = downsample(img, threshold, mask, factor)
photonsD = convert2Photons(imgD, weightD, avgADU)

time3 = time.time()

print "photonsD time: ", time3-time2

pg.image(img, title="LCLS images",levels=(0,200))
#pg.image(photons, title="LCLS photons",levels=(0,30))
#pg.image(mask, title="LCLS mask",levels=(0,1.1))
#pg.image(downsampledImg, title="Downsampled images",levels=(0,200*factor**2))
pg.image(photonsD.astype('float64'), title="Downsampled photons",levels=(0,30))

tic = time.time()
f = h5py.File('img.h5','w')
dset = f.create_dataset("data", img.shape, chunks=(img.shape[0], img.shape[1]))
dset[:,:] = img
f.close()
toc = time.time()
print "img time: ", toc-tic

tic = time.time()
f = h5py.File('img_gzip7.h5','w')
dset = f.create_dataset("data", img.shape, chunks=(img.shape[0], img.shape[1]), compression="gzip", compression_opts=7)
dset[:,:] = img
f.close()
toc = time.time()
print "img_gzip time: ", toc-tic

tic = time.time()
f = h5py.File('imgD_gzip7.h5','w')
dset = f.create_dataset("data", imgD.shape, chunks=(imgD.shape[0], imgD.shape[1]), compression="gzip", compression_opts=7)
dset[:,:] = downsampledImg
f.close()
toc = time.time()
print "imgD_gzip time: ", toc-tic

tic = time.time()
f = h5py.File('photons.h5','w')
dset = f.create_dataset("data", photons.shape, chunks=(photons.shape[0], photons.shape[1]), dtype=np.int16)
dset[:,:] = photons.astype(np.int16)
f.close()
toc = time.time()
print "photons time: ", toc-tic

tic = time.time()
f = h5py.File('photons_gzip7.h5','w')
dset = f.create_dataset("data", photons.shape, chunks=(photons.shape[0], photons.shape[1]), dtype=np.int16, compression="gzip", compression_opts=7)
dset[:,:] = photons.astype(np.int16)
f.close()
toc = time.time()
print "photons_gzip time: ", toc-tic

tic = time.time()
f = h5py.File('photonsD'+str(factor)+'_gzip7.h5','w')
dset = f.create_dataset("data", photonsD.shape, chunks=(photonsD.shape[0], photonsD.shape[1]), dtype=np.int16, compression="gzip", compression_opts=7)
dset[:,:] = photonsD.astype(np.int16)
f.close()
toc = time.time()
print "photonsD_gzip time: ", toc-tic

tic = time.time()
f = h5py.File('img_uint16_gzip7.h5','w')
dset = f.create_dataset("data", img.shape, chunks=(img.shape[0], img.shape[1]), dtype=np.uint16, compression="gzip", compression_opts=7)
_img = img.copy()
_img[np.where(_img<0)]=0
dset[:,:] = _img.astype(np.uint16)
f.close()

# Read in images
tic = time.time()
f = h5py.File('img.h5')
_photonsD = f["data"].value
f.close()
toc = time.time()
print "img read time: ", toc-tic

tic = time.time()
f = h5py.File('img_gzip7.h5')
_photonsD = f["data"].value
f.close()
toc = time.time()
print "img_gzip read time: ", toc-tic

tic = time.time()
f = h5py.File('imgD_gzip7.h5')
_photonsD = f["data"].value
f.close()
toc = time.time()
print "imgD_gzip read time: ", toc-tic

tic = time.time()
f = h5py.File('photons.h5')
_photonsD = f["data"].value
f.close()
toc = time.time()
print "photons read time: ", toc-tic

tic = time.time()
f = h5py.File('photons_gzip7.h5')
_photonsD = f["data"].value
f.close()
toc = time.time()
print "photons_gzip read time: ", toc-tic

tic = time.time()
f = h5py.File('photonsD'+str(factor)+'_gzip7.h5')
_photonsD = f["data"].value
f.close()
toc = time.time()
print "photonsD_gzip read time: ", toc-tic

f = h5py.File('img_uint16_gzip7.h5')
_imgUint16 = f["data"].value
f.close()

# Calculate megabytes
print img.size*32/8/1e6, ' MB' # 32-bit float, 8 bits per byte, 1e6 bytes in MB
print np.max(photonsD), np.max(img)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
