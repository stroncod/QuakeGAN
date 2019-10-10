import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap   
import os
from PIL import ImageChops
import glob
import tqdm
mpl.use('Agg')
plt.ioff()

TIME_STEP = 9
IMG_SIZE = 32
TIME_WND = 7 #IN DAYS
dirpath = os.getcwd()

def init_latlon(data):
    lat_max = data['Latitud'].max()
    lat_min = data['Latitud'].min()
    lon_max = data['Longitud'].max()
    lon_min = data['Longitud'].min()
    lat_m = data['Latitud'].mean()
    lon_m = data['Longitud'].mean()
    
    return lat_max,lat_min,lon_max,lon_min,lat_m,lon_m
def build_time_series(data,y_col=0):
    FIRST = data.Date[:1].values[0]
    dim_0 = data.shape[0]- TIME_WND
    
    y = []
    series = []
    X = []
    k = 0
    for i in range(dim_0):
        series_df = data[(data.Date >= FIRST+i) & (data.Date <= FIRST+i+TIME_WND)]
        k+=1
        #print('creating x')
        #print(FIRST+i, FIRST+i+TIME_WND)
        series.append(series_df)
        if(k==TIME_STEP):
            #print('creating y')
            #print(FIRST+i+TIME_WND)
            #print(FIRST+i+TIME_WND*2)
            y_df = data[(data.Date > FIRST+i+TIME_WND) & (data.Date < FIRST+i+TIME_WND*2)]
            
            #print('len series',len(series))
            X.append(series)
            y.append(y_df)
            series = []
            k = 0

    return X,y


def get_marker_color(magnitude):
    # Returns green for small earthquakes, yellow for moderate
    #  earthquakes, and red for significant earthquakes.
    if magnitude < 3.0:
        return ('go')
    elif magnitude < 5.0:
        return ('yo')
    else:
        return ('ro')

def create_pre_images_x(X):
    
    for i, serie in enumerate(X):
        print(i)
        series_path = dirpath+'/images/pre_images/series_'+str(i)+'_'
        for j, data in enumerate(serie):
            fig = plt.figure(figsize=(8, 8))
            m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                        resolution='i',projection='lcc',lat_0=lat_m,lon_0=lon_m)
            m.shadedrelief()
            m.drawcoastlines(color='gray')
            m.drawcountries(color='gray')
            
            if not data.empty:
                
                lat = data['Latitud'].values
                lon = data['Longitud'].values
                magn = data['Magn'].values
                ##print(magn)
                for lon, lat, mag in zip(lon, lat, magn):
                    x,y = m(lon, lat)

                    marker_string = get_marker_color(mag)
                    m.plot(x, y, marker_string, markersize=10)
            

            plt.savefig(series_path+'Xquakemap_'+str(j)+'.jpg')
            plt.close()
    
def create_pre_images_y(y):
   
    for k, data in enumerate(y):
        print(k)
        series_path =  dirpath+'/images/pre_output/series_'+str(k)+'_'
        fig = plt.figure(figsize=(8, 8))
        m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
            resolution='i',projection='cass',lat_0=lat_m,lon_0=lon_m)
        #m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        if not data.empty:
            lat = data['Latitud'].values
            lon = data['Longitud'].values
            magn = data['Magn'].values
            for lon, lat, mag in zip(lon, lat, magn):
                x,y = m(lon, lat) 
                marker_string = get_marker_color(mag)
                m.plot(x, y, marker_string, markersize=10)
        plt.savefig(series_path+'quakemap.jpg')
        plt.close()
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def gallery(array, ncols=3):

    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)


    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def check_shape(array):
    series = []
    og_size = array[0].size
    for image in array:
        if image.size != og_size:
            im = image.resize(og_size)
            series.append(np.asarray(im))
        else:
           series.append(np.asarray(image)) 
    return series


def make_array(dim_x):
    
    for i in range(dim_x):
        print('series',i) 
        series = []
        aux = []
        for j in range(TIME_STEP):
            print('photo',j)
            aux.append(Image.open(dirpath+'/images/pre_images/series_{0}_Xquakemap_{1}.jpg'.format(i,j)).convert('RGB'))
            
            
        series = check_shape(aux)    
        
        #print(np.shape(series[1]))
       
        im = Image.fromarray(gallery(np.array(series)))
        im.save(dirpath+'/images/gallery/series_{0}_quakemap.jpg'.format(i))
        im_shape = (im.size)

    print(im_shape) 

def triming (namepath):
    folderName = dirpath+namepath
    filePaths = glob.glob(folderName + "/*.jpg")


    for filePath in filePaths:
        #print(filePath)
        image=Image.open(filePath)
        im = trim(image)
        im.save(filePath)

dataset = pd.read_csv('data_earthquake_2000-2017.csv',encoding='latin1', sep=';')

dataset = dataset[dataset['Prof.'] >= 60.0 ]

zone1 = dataset[dataset['Latitud'] > -23]
zone2 = dataset[(dataset.Latitud < -23) & (dataset.Latitud > -34)] 
zone3 = dataset[dataset['Latitud'] < -34]

print('Creating Time Series')
X,y = build_time_series(zone1)

lat_max,lat_min,lon_max,lon_min,lat_m,lon_m = init_latlon(zone1)

print('Creating x photo')
create_pre_images_x(X)
print('Creating Y photo')
create_pre_images_y(y)

print('Triming photos')
triming('/images/pre_images')
triming('/images/pre_output')

print('Create gallery...')

make_array(len(X))

    
    

