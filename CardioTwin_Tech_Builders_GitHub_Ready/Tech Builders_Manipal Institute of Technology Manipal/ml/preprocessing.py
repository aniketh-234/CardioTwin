import numpy as np


class Preprocessor:
    def fit(self,x):
        a=x.reshape(-1,x.shape[-1]).astype(float)
        self.median=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in a.T])
        a=np.where(np.isfinite(a),a,self.median)
        self.mean=a.mean(0); self.scale=a.std(0); self.scale[self.scale<1e-6]=1
        return self

    def transform(self,x):
        missing=~np.isfinite(x)
        values=(np.where(missing,self.median,x)-self.mean)/self.scale
        return np.concatenate([values,missing.astype(float)],axis=-1).astype('float32')

    def dump(self):return {k:getattr(self,k).tolist() for k in ['median','mean','scale']}

    @classmethod
    def load(cls,obj):
        p=cls()
        for k,v in obj.items():setattr(p,k,np.asarray(v))
        return p
