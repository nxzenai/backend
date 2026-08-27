"""
NxZen AI Studio

AutoDL Validators
"""

from __future__ import annotations
from app.modules.autodl.constants import Modality, DLArchitecture
from app.modules.autodl.exceptions import InvalidArchitectureError
from app.modules.autodl.schemas import AutoDLJobCreateRequest

def validate_dl_request(request: AutoDLJobCreateRequest) -> None:
    arch = request.architecture
    mod = request.modality
    
    if mod == Modality.IMAGE and arch != DLArchitecture.CNN:
        raise InvalidArchitectureError(f"Architecture {arch.value} is not suitable for IMAGE modality.")
        
    if mod == Modality.TIME_SERIES and arch != DLArchitecture.RNN:
        raise InvalidArchitectureError(f"Architecture {arch.value} is not suitable for TIME_SERIES modality.")
