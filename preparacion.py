import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder

class PreparadorAIRHSP(TransformerMixin, BaseEstimator):
    """Aprende todas las estadísticas en fit; nunca sobre validación o test."""
    def fit(self,X,y=None):
        self.cols_=list(X.columns)
        self.num_=[c for c in X if c.endswith('_PER_CAPITA') or c in ['CANTIDAD_TOTAL','MES_SIN','MES_COS']]
        self.freq_=[c for c in ['PLIEGO','UNIDAD_EJECUTORA','DESC_CARGO_ESTRUCUTURAL'] if c in X]
        self.cat_=[c for c in X if c not in self.num_+self.freq_+['ESTADO_REGISTRO']]
        self.estado_='ESTADO_REGISTRO' in X
        if self.num_:
            z=X[self.num_].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            self.med_=z.median().fillna(0);self.scaler_=RobustScaler().fit(z.fillna(self.med_))
        self.maps_={c:X[c].fillna('SIN_DATO').astype(str).value_counts(normalize=True).to_dict() for c in self.freq_}
        if self.cat_:
            self.ohe_=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
            self.ohe_.fit(X[self.cat_].fillna('SIN_DATO').astype(str))
        names=self.num_+[c+'_FREQ' for c in self.freq_]
        if self.cat_:names+=list(self.ohe_.get_feature_names_out(self.cat_))
        if self.estado_:names+=['ESTADO_REGISTRO_VACANTE']
        self.names_=np.array([f'f{i:04d}_'+re.sub('[^0-9A-Za-z_]','_',c) for i,c in enumerate(names)])
        return self
    def transform(self,X):
        if set(X.columns)!=set(self.cols_):raise ValueError('Esquema distinto al de fit.')
        parts=[]
        if self.num_:
            z=X[self.num_].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(self.med_)
            parts.append(self.scaler_.transform(z).astype(np.float32))
        for c in self.freq_:parts.append(X[c].fillna('SIN_DATO').astype(str).map(self.maps_[c]).fillna(0).to_numpy(np.float32)[:,None])
        if self.cat_:parts.append(self.ohe_.transform(X[self.cat_].fillna('SIN_DATO').astype(str)))
        if self.estado_:
            if not X.ESTADO_REGISTRO.isin(['OCUPADO','VACANTE']).all():raise ValueError('Estado actual inválido.')
            parts.append(X.ESTADO_REGISTRO.eq('VACANTE').to_numpy(np.float32)[:,None])
        out=np.concatenate(parts,axis=1)
        if not np.isfinite(out).all():raise ValueError('Valores no finitos tras preparación.')
        return out
    def get_feature_names_out(self,input_features=None):return self.names_.copy()
