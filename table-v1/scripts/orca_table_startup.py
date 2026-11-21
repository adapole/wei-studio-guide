"""Causal startup command blending from the first measured hand state, without reference commands."""
import numpy as np


class MeasuredStartup:
    def __init__(self,policy,seconds=1.):
        if not np.isfinite(seconds) or not 0<seconds<=2:raise ValueError('Startup duration must be in (0,2] seconds')
        self.policy=policy;self.seconds=seconds;self.reset()
    def reset(self):
        self.policy.reset();self.tick=0;self.initial=None;self.events=[];self.raw_native=[]
    def __call__(self,observation):
        if self.initial is None:
            state=np.asarray(observation['observation.state'],dtype=float)
            if state.shape!=(47,) or not np.isfinite(state).all():raise ValueError('Expected measured native hand state')
            self.initial=np.r_[state[:17],state[34:41]].copy()
            norm=np.linalg.norm(self.initial[20:24])
            if norm<1e-12:raise ValueError('Degenerate measured orientation')
            self.initial[20:24]/=norm
        raw=np.asarray(self.policy(observation),dtype=float).copy()
        if raw.shape!=(24,) or not np.isfinite(raw).all():raise ValueError('Policy must predict a finite native command')
        if np.linalg.norm(raw[17:20])>2:raise ValueError('Policy base target exceeds local workspace')
        if np.linalg.norm(raw[20:24])<1e-6:raise ValueError('Degenerate predicted orientation')
        self.raw_native.append(raw.copy())
        elapsed=self.tick/60;self.tick+=1
        if elapsed>=self.seconds:return raw
        alpha=float(.5-.5*np.cos(np.pi*elapsed/self.seconds));result=(1-alpha)*self.initial+alpha*raw
        q=raw[20:24].copy();norm=np.linalg.norm(q)
        if norm<1e-6:raise ValueError('Degenerate predicted orientation')
        q/=norm
        if self.initial[20:24]@q<0:q=-q
        result[20:24]=(1-alpha)*self.initial[20:24]+alpha*q;result[20:24]/=np.linalg.norm(result[20:24])
        self.events.append({'command_tick':self.tick-1,'time_s':elapsed,'blend_fraction':alpha,'raw_decoded_policy_action':raw.tolist(),'requested_after_startup':result.tolist()})
        return result
    def report(self):
        return {'recipe':'measured-first-observation-cosine-startup-v1','seconds':self.seconds,'model_calls':self.tick,'initial_measured_hand_command':None if self.initial is None else self.initial.tolist(),
            'modified_command_count':len(self.events),'events':self.events,'reference_commands_used':False,'object_state_modified':False}
