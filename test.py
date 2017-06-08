#!/usr/local/bin/python

import numpy as np
import h5py as hpy
import warnings
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap,cm

# Set the lag day for the temperature/index relationship.
# lagDay is the number of days AFTER the indices that the temperature matrix begins.
# For a weeks 3/4 forecast, this should be around 15.
lagDay = 15

AO = np.loadtxt('/Users/kyle/Downloads/AOindex_start1981.csv')
NAO = np.loadtxt('/Users/kyle/Downloads/NAOindex_start1981.csv')
PNA = np.loadtxt('/Users/kyle/Downloads/PNAindex_start1981.csv')

matFile = hpy.File('/Users/kyle/Downloads/testGridPoint.mat', 'r')

tempMatrix = matFile.get('test') # Dimensions of time x lat x lon. Python reads in as 1 x 13149

# Normalize all the data.
AO = stats.zscore(AO)
NAO = stats.zscore(NAO)
PNA = stats.zscore(PNA)

tempMatrix = tempMatrix[0,lagDay:AO.size+lagDay] # This makes the matrix begin 15 days after the indices.

lats = np.arange(89,-90,-1)
lons = np.arange(0,360,1) # remember that matrix above is 0 to 359.

tempMatrixZ = stats.zscore(tempMatrix)

# Put all the indices together
indices = np.vstack((AO[0:12426], NAO[0:12426], PNA[0:12426]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4,input_shape=(1,1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Need to get these into order: <variable (sample) x timestep x 1>
indicesModel = np.reshape(indices[0,:],(indices.shape[1],1,1))
tempData = tempMatrixZ[1:indices.shape[1]+1]

model.fit(indicesModel, tempData, nb_epoch=10, batch_size=1, verbose=2)

scores = model.evaluate(indicesModel, tempData[1:indices.shape[1]+1])
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

########################################################################################################
## Plot figure, define geographic bounds
#fig = plt.figure(dpi=200)
#latCorners = ([20,90])
#lonCorners = ([190,340])
#
#m = Basemap(projection='cyl',llcrnrlat=latCorners[0],urcrnrlat=latCorners[1],llcrnrlon=lonCorners[0],urcrnrlon=lonCorners[1])
#
## Draw coastlines, political boundaries
#m.drawcoastlines()
#m.drawstates()
#m.drawcountries()
#
## Draw filled contours.
#clevs = np.arange(260,360,5)
#
#lats = np.arange(80,19,-1)
#lons = np.arange(190,341,1) # remember that matrix above is 0 to 359.
#
## Define the lat and lon data
#x,y = np.int16(np.meshgrid(lons,lats))
#cs = m.contourf(x,y,tempMatrix[:,:,1].T,cmap=cm.GMT_drywet,latlon=True)
#
## Set the title and fonts
#plt.title('Sample Temperature Map')
#
##  Add colorbar
#cbar = m.colorbar(cs,location='right',pad="5%")
#cbar.set_label('K')
#plt.savefig('testTemperatureMap.png',dpi=300)-