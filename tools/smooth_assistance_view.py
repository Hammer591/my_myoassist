"""Display-only smoothing and descriptive diagnostics for saved hip moments."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/myoassist_mpl')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


def periodic_smooth(cycles, sigma):
    # 0% and 100% are the same point: omit duplicate endpoint while filtering.
    sm = gaussian_filter1d(cycles[:, :-1, :], sigma=sigma, axis=1, mode='wrap')
    return np.concatenate([sm, sm[:, :1, :]], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    args = ap.parse_args()
    p = args.input
    out = p/'smoothed_view'
    out.mkdir(exist_ok=True)
    source = json.loads((p/'summary.json').read_text())
    phase = np.linspace(0, 100, 201)
    metrics = {'smoothing': {'scope': 'Display only; model, simulation and source data unchanged',
        'cycle': 'Periodic Gaussian; sigma=2 percentage points of the gait cycle; each stride filtered before averaging',
        'time': '4th-order Butterworth; 6 Hz cutoff; forward-backward zero-phase; 1200 Hz source',
        'sd': 'Across-stride standard deviation after smoothing; not a confidence interval'}, 'sides': {}}
    fig, axes = plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    full, fax = plt.subplots(2,2,figsize=(11,7),sharex=True,sharey='row')
    table, headers = [phase], ['gait_cycle_pct']
    for i,side in enumerate('rl'):
        raw = np.load(p/f'gait_cycles_{side}.npy')
        sm = periodic_smooth(raw, sigma=4)
        mean,sd = sm.mean(axis=0), sm.std(axis=0)
        original = raw.mean(axis=0)
        color = ['#176eaa','#ce6230'][i]
        for ax in [axes[i],fax[0,i]]:
            ax.plot(phase,original[:,0],color='0.65',lw=.9,label='Original mean',alpha=.8)
            ax.fill_between(phase,mean[:,0]-sd[:,0],mean[:,0]+sd[:,0],color=color,alpha=.16)
            ax.plot(phase,mean[:,0],color=color,lw=2.2,label='Smoothed mean +/- 1 SD')
            ax.set_title(f'{"Right" if side=="r" else "Left"} hip assistance (n={len(raw)} strides)')
            ax.set(xlim=(0,100),xlabel='Own-leg gait cycle (%)',ylabel='Torque (Nm); + flexion / - extension')
            ax.legend(fontsize=8)
            ax.axhline(0,color='0.4',lw=.7)
            ax.grid(alpha=.2)
        ax=fax[1,i]
        ax.plot(phase,original[:,1],color=color,lw=1.6)
        bsd=raw[:,:,1].std(axis=0)
        ax.fill_between(phase,original[:,1]-bsd,original[:,1]+bsd,color=color,alpha=.16)
        ax.set(title='Muscle-only biological hip moment (original)',xlabel='Own-leg gait cycle (%)',ylabel='Moment (Nm); + flexion / - extension',xlim=(0,100))
        ax.axhline(0,color='0.4',lw=.7)
        ax.grid(alpha=.2)
        unwrapped=original[:-1,0]
        m=mean[:-1,0]
        ph=phase[:-1]
        metrics['sides'][side]={'n_strides':len(raw),'smoothed_flexion_peak_Nm':float(m.max()),
            'smoothed_flexion_peak_phase_pct':float(ph[m.argmax()]),
            'smoothed_extension_peak_Nm':float(m.min()),'smoothed_extension_peak_phase_pct':float(ph[m.argmin()]),
            'raw_flexion_peak_Nm':float(unwrapped.max()),'raw_flexion_peak_phase_pct':float(ph[unwrapped.argmax()]),
            'raw_extension_peak_Nm':float(unwrapped.min()),'raw_extension_peak_phase_pct':float(ph[unwrapped.argmin()]),
            'raw_same_sign_impulse_fraction':float((abs(unwrapped)*(unwrapped*original[:-1,1]>0)).sum()/abs(unwrapped).sum()),
            'phase_method':{},'sigma_sensitivity':{}}
        for sigma in [1,2,3]:
            v=periodic_smooth(raw,sigma=sigma*2).mean(axis=0)[:-1,0]
            metrics['sides'][side]['sigma_sensitivity'][str(sigma)]={'flexion_peak_phase_pct':float(ph[v.argmax()]),'extension_peak_phase_pct':float(ph[v.argmin()])}
        table.extend([original[:,0],mean[:,0],sd[:,0]])
        headers.extend([f'{side}_original_mean_Nm',f'{side}_smoothed_mean_Nm',f'{side}_smoothed_sd_Nm'])
    for figure,name in [(fig,'assistance_smoothed'),(full,'hip_moments_smoothed_assistance')]:
        figure.suptitle('PhotoHipExo | final checkpoint 30,015,488 | assistance ON\nDisplay-only smoothing: periodic Gaussian, sigma = 2% gait cycle')
        figure.tight_layout()
        for ext in ['png','pdf']:
            figure.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(figure)
    np.savetxt(out/'assistance_smoothed.csv',np.column_stack(table),delimiter=',',header=','.join(headers),comments='')
    first=np.load(p/'hip_moments_seed1234.npz')['data']
    sos=butter(4,6,fs=1200,output='sos')
    filtered=sosfiltfilt(sos,first[:,1:3],axis=0)
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,sharey=True)
    for i,ax in enumerate(axes):
        ax.plot(first[:,0],first[:,1+i],color='0.7',lw=.8,label='Actual torque (original)')
        ax.plot(first[:,0],filtered[:,i],color=['#176eaa','#ce6230'][i],lw=1.7,label='Display: 6 Hz zero-phase low-pass')
        ax.set(ylabel=f'{"Right" if i==0 else "Left"} hip torque (Nm)',xlim=(2,7))
        ax.axhline(0,color='0.4',lw=.7)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Elapsed time (s)')
    fig.suptitle('Display-only smoothing | seed 1234 | + flexion / - extension')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/f'assistance_timeseries_smoothed.{ext}',dpi=180)
    plt.close(fig)
    np.savetxt(out/'assistance_timeseries_smoothed.csv',np.column_stack([first[:,0],first[:,1:3],filtered]),delimiter=',',header='time_s,raw_r_Nm,raw_l_Nm,display_r_Nm,display_l_Nm',comments='')
    # Power diagnostic from unsmoothed, synchronous 1200 Hz physics samples.
    power=[]
    for rollout in source['rollouts']:
        seed=rollout['seed']
        gait=json.loads((p/f'gait_seed{seed}.json').read_text())
        physics=np.load(p/f'hip_moments_seed{seed}.npz')['data']
        for i,side in enumerate('rl'):
            w=physics[:,9+i]
            np.testing.assert_allclose(w,physics[:,1+i]*physics[:,7+i],atol=1e-10)
            segments=rollout['segments'][side]
            profiles=[]
            for a,t,b in segments:
                a,b=(a+1)*40-1,(b+1)*40-1
                profiles.append(np.interp(phase,np.linspace(0,100,b-a+1),w[a:b+1]))
            profiles=np.array(profiles)
            neg=np.maximum(-profiles[:,:-1],0)
            phase_mask=(phase[:-1]>=60)&(phase[:-1]<80)
            power.append({'seed':seed,'side':side,'positive_mean_W':float(np.maximum(w,0).mean()),
                'negative_mean_W':float(np.minimum(w,0).mean()),
                'negative_phase60_80_share':float(neg[:,phase_mask].sum()/neg.sum())})
            # Identify whether the saved phase segments use the hip-peak fallback.
            from scipy.signal import find_peaks
            hip=np.array(gait['series_data']['joint_data'][f'hip_flexion_{side}']['qpos']).ravel()
            peaks,_=find_peaks(hip,distance=15,prominence=.15)
            strikes=np.round(peaks+.07*np.diff(peaks).mean()).astype(int)
            fallbacks={(int(a),int(a+round(.6*(b-a))),int(b)) for a,b in zip(strikes[:-1],strikes[1:])}
            metrics['sides'][side]['phase_method'][str(seed)]='hip-peak fallback (approximate heel strike)' if all(tuple(seg) in fallbacks for seg in segments) else 'foot-contact segmentation'
    metrics['power_diagnostic_1200Hz']=power
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
