# Installation

You need Python 3.10 or newer and git. Linux and macOS are tested; the pipeline is plain Python and MuJoCo, so Windows should work as well (the shell script `pipeline/run_all.sh` needs bash; the `mocap2myo` command does not).

```bash
git clone https://github.com/Fitz-Yuan/mocap-to-myofullbody
cd mocap-to-myofullbody
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[c3d,tak]"
mocap2myo --help                     # the command is installed
```

Install from a clone (`pip install -e .`): the command runs the repository's scripts (`pipeline/`, `tasks/`).

## Optional parts

| Extra | For | Installs |
|---|---|---|
| `c3d` | reading C3D files (markers, force plates, analog channels) | ezc3d |
| `tak` | reading OptiTrack Motive `.tak` files | olefile |
| `b3d` | reading AddBiomechanics `.b3d` files | nimblephysics |
| `gpu` | the IK on an NVIDIA GPU (`--gpu`, `M2M_IK_DEVICE=gpu`) | torch, warp-lang |
| `dev` | running the tests | pytest, ruff |

For example `pip install -e ".[c3d,tak,gpu]"`. The OpenSim comparison (`run --opensim`) also needs the `opensim` package from conda (`conda install -c opensim-org opensim`), in this or another Python (`--osim-python`).

## Models

MyoFullBody, the default model, is in the repository. The other models are fetched at pinned versions and converted to MuJoCo where needed:

```bash
mocap2myo models                     # all of them
mocap2myo models rajagopal2016       # one
```

They go into `M2M_MODELS_DIR` (default `~/data/mocap2myo/models`) and keep their own licences ({doc}`../reference/license`). Converting the OpenSim models needs MyoConverter's environment; `models/fetch.py --help` explains it.

## Check the installation

```bash
pip install -e ".[dev]"
pytest -q tests
```
