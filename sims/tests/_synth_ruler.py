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


def synth_card(rng, pitch=53.33, blur=1.3, tilt=1.0, H=480, W=640, cx=331.7, cy=257.3, line=2.2, small=22.0, long=38.0, noise=2.0):
    """The Captrail calibration card at nail zoom: a dark crosshair, thin fine ticks across both arms
    (100 um apart), a longer tick every 10, slight tilt, vignetting and noise. Returns a grey image."""
    up = 4
    yy, xx = np.mgrid[0:H * up, 0:W * up] / up
    t = np.deg2rad(tilt)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)

    def ticks(a, b):
        k = np.round(a / pitch)
        return (np.abs(a - k * pitch) < line / 2) & (np.abs(b) < np.where(k % 10 == 0, long, small))
    dark = (np.abs(v) < line / 2) | (np.abs(u) < line / 2) | ticks(u, v) | ticks(v, u)
    img = np.where(dark, 60.0, 205.0).reshape(H, up, W, up).mean((1, 3))
    img = ndi.gaussian_filter(img, blur)
    vy, vx = np.mgrid[0:H, 0:W]
    img *= 1 - 0.2 * (((vx - 320) / 400) ** 2 + ((vy - 240) / 300) ** 2)
    img += rng.normal(0, noise, img.shape)
    return img.clip(0, 255)


def synth_white_card_nailzoom(rng, H=480, W=640):
    """A white card at nail zoom: strong vignetting and soft paper blotches, so it fails the strict blank test
    (raw SD > 15) but has almost no fine structure."""
    vy, vx = np.mgrid[0:H, 0:W]
    img = 200 * (1 - 0.35 * (((vx - 320) / 380) ** 2 + ((vy - 240) / 300) ** 2))
    img += 2.5 * ndi.gaussian_filter(rng.normal(size=(H, W)), 8) * 8
    img += rng.normal(0, 1.0, (H, W))
    return img.clip(0, 255)
