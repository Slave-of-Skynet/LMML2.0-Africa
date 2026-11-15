"""Ethiopia: spectral-valley recording profiles with fine splice-boundary fit.

Reads the WAV filename from argv[1]; writes only detection.csv in the working
directory. No reference data or external model is used. NumPy and SciPy only.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import csv
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

FRAME, HOP, BANDS = 4096, 1024, 48


def spectrum(x, sr):
    """Estimate a recording-noise profile from spectral valleys, not harmonics."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1: x=x[:,None]
    x=x-np.mean(x,axis=0,keepdims=True)
    # Preserve high-frequency recording/codec fingerprints at common WAV rates.
    if sr>48000:
        divisor=gcd(int(sr),48000)
        x=resample_poly(x,48000//divisor,int(sr)//divisor,axis=0)
        sr=48000
    if len(x)<FRAME:
        x=np.pad(x,((0,FRAME-len(x)),(0,0)))
    window=np.hanning(FRAME)
    count=1+(len(x)-FRAME)//HOP
    power=np.zeros((count,FRAME//2+1))
    for channel in range(x.shape[1]):
        frames=np.lib.stride_tricks.sliding_window_view(x[:,channel],FRAME)[::HOP]
        # Bounded FFT temporaries, including stereo/high-rate input.
        for first in range(0,count,128):
            transformed=np.fft.rfft(frames[first:first+128]*window,axis=1)
            power[first:first+128]+=abs(transformed)**2/x.shape[1]
    frequencies=np.fft.rfftfreq(FRAME,1/sr)
    edges=np.geomspace(min(30,sr/8),sr/2,BANDS+1)
    eps=max(float(power.max())*1e-12,1e-20)
    floor=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        band=power[:,(frequencies>=lo)&(frequencies<hi)]
        if band.shape[1]:
            # Sparse musical peaks can dominate a sum while leaving valleys.
            value=np.percentile(band,20,axis=1)
            floor.append(10*np.log10(value+eps))
        else:
            floor.append(np.full(count,10*np.log10(eps)))
    return np.stack(floor,axis=1),sr


def aggregate(db,sr,window=1.0,step_s=.23):
    w=max(1,min(len(db),round(window*sr/HOP)))
    stride=max(1,round(step_s*sr/HOP))
    starts=np.arange(0,len(db)-w+1,stride)
    windows=np.lib.stride_tricks.sliding_window_view(db,w,axis=0)[starts]
    raw=np.concatenate([np.median(windows,axis=-1),np.percentile(windows,10,axis=-1)],axis=1)
    centers=(starts*HOP+((w-1)*HOP+FRAME)/2)/sr
    return centers,raw,stride*HOP/sr


def representation(raw,whiten=True,ridge_floor=.1):
    center=np.median(raw,axis=0)
    scale=1.4826*np.median(abs(raw-center),axis=0)+.5
    z=np.clip((raw-center)/scale,-8,8)
    if not whiten:
        return z,(center,scale,np.eye(z.shape[1]))
    delta=np.diff(z,axis=0)
    delta=np.clip(delta,-3,3)
    covariance=delta.T@delta/max(1,len(delta))
    ridge=max(ridge_floor,float(np.trace(covariance))/len(scale)*.1)
    values,vectors=np.linalg.eigh(covariance)
    transform=(vectors/np.sqrt(np.maximum(values,0)+ridge))@vectors.T
    return z@transform,(center,scale,transform)


def transform_raw(raw,parameters):
    center,scale,transform=parameters
    return np.clip((raw-center)/scale,-8,8)@transform


def candidates(z,step,mode='global',count=5,min_s=1.0,max_s=None):
    n=len(z)
    if n<3: return [(0,max(1,n),0.)]
    p=np.vstack([np.zeros(z.shape[1]),np.cumsum(z,axis=0)])
    minimum=max(2,int(np.ceil(min_s/step)))
    minimum=min(minimum,n-1)
    maximum=min(n-1,int(max_s/step) if max_s else max(minimum,n//2))
    maximum=max(maximum,minimum)
    proposals=[]
    for i in range(n-minimum+1):
        j=np.arange(i+minimum,min(n,i+maximum)+1)
        inside=(p[j]-p[i])/(j-i)[:,None]
        valid=(j-i)<n
        j,inside=j[valid],inside[valid]
        outside=(p[n]-p[j]+p[i])/(n-j+i)[:,None]
        scores=np.sum((inside-outside)**2,axis=1)*(j-i)*(n-j+i)/n
        if len(j):
            top=np.argsort(scores)[-min(3,len(j)):]
            proposals.extend((i,int(j[k]),float(scores[k])) for k in top)
    proposals.sort(key=lambda v:-v[2])
    selected=[]
    for proposal in proposals:
        i,j,score=proposal
        if all(max(0,min(j,b)-max(i,a))/max(j-i,b-a)<.6 for a,b,_ in selected):
            selected.append(proposal)
            if len(selected)==count: break
    return selected


def interval_times(t,i,j,duration):
    left=0. if i==0 else (t[i-1]+t[i])/2
    right=duration if j==len(t) else (t[j-1]+t[j])/2
    return float(max(0,left)),float(min(duration,right))


def refine(db,sr,t,z,proposal,parameters,duration):
    i,j,score=proposal
    a,b=interval_times(t,i,j,duration)
    fine_t,fine_raw,_=aggregate(db,sr,.35,HOP/sr)
    fine=transform_raw(fine_raw,parameters)
    interior=(fine_t>a+.4)&(fine_t<b-.4)
    if interior.sum()<3: return a,b
    mu_in=np.median(fine[interior],axis=0)
    left=(fine_t>=max(0,a-3))&(fine_t<a-.4)
    right=(fine_t>b+.4)&(fine_t<=min(duration,b+3))
    # An edge-touching coarse proposal may include clean leading/trailing
    # audio. Learn its inserted regime next to the other, supported join.
    if left.sum()<3 and right.sum()>=3:
        core=(fine_t>=max(a+.2,b-1.5))&(fine_t<b-.4)
        if core.sum()>=3: mu_in=np.median(fine[core],axis=0)
    elif right.sum()<3 and left.sum()>=3:
        core=(fine_t>a+.4)&(fine_t<=min(b-.2,a+1.5))
        if core.sum()>=3: mu_in=np.median(fine[core],axis=0)
    estimates=[]
    for boundary,exterior,first in ((a,left,True),(b,right,False)):
        edge=exterior.sum()<3
        if exterior.sum()<3:
            exterior=right if first else left
            if exterior.sum()<3:
                estimates.append(boundary); continue
        mu_out=np.median(fine[exterior],axis=0)
        if edge:
            local=np.flatnonzero((fine_t>=0)&(fine_t<=b-.5)) if first else np.flatnonzero((fine_t>=a+.5)&(fine_t<=duration))
        else:
            local=np.flatnonzero((fine_t>=boundary-1.5)&(fine_t<=boundary+1.5))
        if len(local)<3:
            estimates.append(boundary); continue
        d_in=np.sum((fine[local]-mu_in)**2,axis=1)
        d_out=np.sum((fine[local]-mu_out)**2,axis=1)
        before,after=(d_out,d_in) if first else (d_in,d_out)
        values=np.r_[0,np.cumsum(before)]+np.r_[np.cumsum(after[::-1])[::-1],0]
        k=int(np.argmin(values))
        pos=(fine_t[local[0]] if k==0 else fine_t[local[-1]] if k==len(local)
             else (fine_t[local[k-1]]+fine_t[local[k]])/2)
        estimates.append(float(pos))
    if estimates[1]-estimates[0]<.5: return a,b
    return max(0,estimates[0]),min(duration,estimates[1])


def detect(x,sr,mode='global',whiten=True,refinement=True):
    if sr<=0: raise ValueError('Invalid sample rate')
    duration=len(x)/sr
    if duration<=0: raise ValueError('Empty WAV')
    if not np.isfinite(x).all(): raise ValueError('Non-finite audio samples')
    if np.max(abs(np.asarray(x,dtype=float)))<1e-15: return 0.,duration
    if np.std(np.asarray(x,dtype=float),axis=0).max()<1e-15: return 0.,duration
    db,rate=spectrum(x,sr)
    t,raw,step=aggregate(db,rate)
    z,parameters=representation(raw,whiten)
    options=candidates(z,step,mode)
    if not options: return 0.,duration
    if refinement:
        return refine(db,rate,t,z,options[0],parameters,duration)
    return interval_times(t,*options[0][:2],duration)


def main():
    sr,x=wavfile.read(sys.argv[1])
    start,end=detect(x,sr)
    with open('detection.csv','w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['start_s','end_s'])
        writer.writerow([f'{start:.3f}',f'{end:.3f}'])


if __name__=='__main__': main()
