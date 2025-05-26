"""QSAR model scorer for REINVENT 4 (DRD2 RF model)"""

__all__ = ["drd2rfqsar"]
from dataclasses import dataclass
from typing import List
import os
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

from .component_results import ComponentResults
from .add_tag import add_tag

from reinvent_plugins.mol_cache import molcache

# ----------------------------- Parameters -----------------------------

@add_tag("__parameters")
@dataclass
class Parameters:
    model_path: List[str]  # path to .pkl model

# ----------------------------- QSAR component -----------------------------

@add_tag("__component")
class DRD2RFQSAR:
    def __init__(self, params: Parameters):
        # Load model from the provided path
        if len(params.model_path) != 1:
            raise ValueError("Provide exactly one model path.")
        
        self.model_file = os.path.join(os.path.dirname(__file__), params.model_path[0])
        print(f"DRD2RFQSAR: Model path {self.model_file}")
        #self.model_file = params.model_path[0]
        self._rf = joblib.load(self.model_file)
        self.N_BITS = 2048
        self.RADIUS = 3

    def mol_to_fp(self, mol):
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.RADIUS, nBits=self.N_BITS)
        return np.asarray(fp, dtype=np.uint8)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        fps = []
        valid_mask = []
        for mol in mols:
            fp = self.mol_to_fp(mol)
            if fp is None:
                valid_mask.append(False)
                fps.append(np.zeros(self.N_BITS, dtype=np.uint8))
            else:
                valid_mask.append(True)
                fps.append(fp)

        fps = np.stack(fps)
        probs = self._rf.predict_proba(fps)[:, 1]
        probs = np.where(valid_mask, probs, 0.0)

        # Wrap in ComponentResults
        return ComponentResults([probs])

