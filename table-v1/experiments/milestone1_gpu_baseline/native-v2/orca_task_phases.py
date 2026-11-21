"""Sampled task diagnostics. Authoritative physical gates remain in the pinned validator."""
class TaskPhases:
    def __init__(self):
        self.phase=None;self.intervals=[];self.events={};self.last=None;self.hold=0.;self.settle=0.;self.ever_lifted=False

    def update(self,time,contact,opposed,lifted,landed,hand_load):
        dt=0. if self.last is None else time-self.last
        if dt<0:raise ValueError('Non-monotonic diagnostic clock')
        self.hold=self.hold+dt if lifted else 0.;self.settle=self.settle+dt if landed else 0.
        self.ever_lifted|=lifted
        events={'first_contact':contact,'first_opposing_support':opposed,'first_supported_lift':lifted,'supported_hold_2s':self.hold>=2,'first_release_after_lift':self.ever_lifted and hand_load<.05,'tray_settle_1s':self.settle>=1}
        for name,active in events.items():
            if active and name not in self.events:self.events[name]=time
        if self.settle>=1:phase='settled'
        elif landed:phase='tray-support'
        elif self.ever_lifted and hand_load<.05:phase='released'
        elif self.hold>=2:phase='holding'
        elif lifted:phase='lifting'
        elif opposed:phase='opposing-grasp'
        elif contact:phase='contact'
        else:phase='approach'
        if phase!=self.phase:
            self.intervals.append({'phase':phase,'start_s':time,'end_s':time});self.phase=phase
        self.intervals[-1]['end_s']=time;self.last=time
        return phase

    def report(self):return {'scope':'sampled diagnostics only; do not replace full-physics acceptance gates','events':self.events,'intervals':self.intervals}
