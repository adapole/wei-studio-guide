"""Physical-unit prediction errors; aggregate loss alone is not a task-success metric."""
import numpy as np
def metrics(predicted,submitted,joint_ids):
    predicted=np.asarray(predicted,dtype=float);submitted=np.asarray(submitted,dtype=float)
    if predicted.shape!=submitted.shape or predicted.ndim!=2 or predicted.shape[1]!=24 or not np.isfinite(predicted).all():raise ValueError('Expected matching finite N x 24 action arrays')
    errors=np.abs(predicted[:,:17]-submitted[:,:17]);position=np.linalg.norm(predicted[:,17:20]-submitted[:,17:20],axis=1);qa=predicted[:,20:24];qb=submitted[:,20:24];denom=np.linalg.norm(qa,axis=1)*np.linalg.norm(qb,axis=1)
    if np.any(denom<1e-6):raise ValueError('Invalid quaternion prediction')
    angles=2*np.arccos(np.clip(np.abs(np.sum(qa*qb,axis=1))/denom,0,1))
    stat=lambda x:{'mean':float(np.mean(x)),'p95':float(np.quantile(x,.95)),'maximum':float(np.max(x))}
    return {'frames':len(predicted),'joint_absolute_error_rad':{name:stat(errors[:,i]) for i,name in enumerate(joint_ids)},'base_position_error_m':stat(position),'base_orientation_error_rad':stat(angles)}
