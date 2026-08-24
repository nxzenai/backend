# Python Lab runtime profile

`requirements-notebook-runtime.txt` defines the optional CPU notebook profile.
It does not enable unrestricted package installation. The runtime information
endpoint reports availability from the process launching the local Jupyter
kernel and reports GPU only after PyTorch or TensorFlow confirms real hardware.

## Verified local environment — 2026-08-24

| Capability | Status |
|---|---|
| Python | 3.14.5 |
| NumPy | 2.4.1 |
| pandas | 2.3.3 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| Seaborn | 0.13.2 |
| scikit-learn | 1.8.0 |
| Jupyter client | 8.8.0 |
| IPython | 9.9.0 |
| PyTorch | 2.10.0 |
| TensorFlow/Keras | Not installed (the profile excludes Python 3.14) |
| transformers | 5.0.0 |
| datasets | Not installed |
| spaCy | 3.8.11 metadata present, import failed on Python 3.14/Pydantic v1 compatibility |
| NLTK | 3.9.2 |
| GPU | Unavailable; framework detection returned false |

Availability is deployment-specific. The UI and API report the live result and
never substitute these development-machine values.
