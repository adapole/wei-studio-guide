"""Explicit command-boundary handling; never changes achieved or object state."""
import numpy as np

class JointCommands:
    def __init__(self,submit,joint_ids,mode='strict'):
        if mode not in ['strict','bounded']:raise ValueError('Unknown joint command mode')
        self.submit=submit;self.ids=joint_ids;self.mode=mode;self.reset()

    def reset(self):self.calls=0;self.events=[]

    def __call__(self,value,ranges):
        raw=np.asarray(value,dtype=float).reshape(-1);limits=np.asarray(ranges,dtype=float);tick=self.calls;self.calls+=1
        if raw.shape!=(24,) or limits.shape!=(17,2) or not np.isfinite(raw).all():return self.submit(value,ranges)
        submitted=raw.copy();submitted[:17]=np.clip(raw[:17],limits[:,0],limits[:,1])
        indices=np.flatnonzero(raw[:17]!=submitted[:17])
        if len(indices):
            details=[{'joint_index':int(i),'joint_id':self.ids[i],'predicted_rad':float(raw[i]),'lower_rad':float(limits[i,0]),'upper_rad':float(limits[i,1]),'submitted_rad':float(submitted[i]) if self.mode=='bounded' else None,'overshoot_rad':float(abs(raw[i]-submitted[i]))} for i in indices]
            self.events.append({'command_tick':tick,'time_s':tick/60,'mode':self.mode,'raw_action':raw.tolist(),'submitted_action':submitted.tolist() if self.mode=='bounded' else None,'joints':details})
            if self.mode=='strict':raise ValueError('Policy joint limits: '+str(details))
        return self.submit(submitted if self.mode=='bounded' else raw,ranges)

    def report(self):
        return {'mode':self.mode,'command_count':self.calls,'limit_events':len(self.events),'max_overshoot_rad':max((j['overshoot_rad'] for e in self.events for j in e['joints']),default=0.),'events':self.events}
