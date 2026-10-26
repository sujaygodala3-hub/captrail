"""Synthetic stage micrometer for tests: fine ticks, long ticks every 10, crosshair, comb, dust, vignetting, tilt."""
import numpy as np
from scipy import ndimage as ndi
def synth_ruler(rng, pitch=16.42, blur=1.2, tilt=0.8, H=480, W=640, x0=7.3, dust=40):
    up=4; h,w=H*up,W*up
    yy,xx=np.mgrid[0:h,0:w]/up
    xr=xx*np.cos(np.deg2rad(tilt))+yy*np.sin(np.deg2rad(tilt))
    ph=(xr-x0)%pitch; k=np.floor((xr-x0)/pitch)
    tick=(ph<pitch*0.5)
    fine=tick&(np.abs(yy-240)<50)
    long_=tick&(k%10==0)&(np.abs(yy-240)<95)
    hline=np.abs(yy-240)<1.5
    cross=(np.abs(xx-300)<3)&(yy<200)
    comb=(np.abs(yy-60)<30)&((xx%9)<4)&(np.abs(xx-300)<120)
    dark=fine|long_|hline|cross|comb
    img=np.where(dark,70.0,200.0)
    img=img.reshape(H,up,W,up).mean((1,3))
    img=ndi.gaussian_filter(img,blur)
    vy,vx=np.mgrid[0:H,0:W]; img*=1-0.25*(((vx-320)/400)**2+((vy-240)/300)**2)
    for _ in range(dust):
        cy,cx=rng.integers(0,H),rng.integers(0,W); r=rng.uniform(1,4)
        img[(vy-cy)**2+(vx-cx)**2<r*r]*=0.6
    img+=rng.normal(0,2,img.shape)
    return np.stack([img]*3,-1).clip(0,255)
