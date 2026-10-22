# The demo

```bash
mocap2myo demo
```

The demo takes one walking trial of a public dataset (Moissenet & Schreiber 2019, Qualisys, 100 Hz, 52 ISB markers, two force plates; CC BY 4.0) and runs every step on it:

1. It downloads the trial and the subject's static trial, about 0.4 MB. Only these two files are read out of the 381 MB archive (by HTTP range requests), into `mocap2myo_demo/raw/`.
2. It runs the pipeline with the dataset's profile (`datasets/moissenet2019.toml`): read the C3D files, prepare the markers, calibrate the subject on the static trial, solve the IK, grade the trial, compute the joint loads with the force plates.
3. It writes a clip for the viewer, `mocap2myo_demo/viewer/2014001_C3_03.elvw`.

It takes about a minute. Open the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/) and drop the `.elvw` file on it: the model walks, its virtual markers sit next to the measured ones (coloured by the fit error), and the panels show the joint angles, the joint moments and the ground reaction. The file is read in the browser; nothing is uploaded.

What the demo wrote:

| Where | What |
|---|---|
| `mocap2myo_demo/markers/` | the conditioned markers |
| `mocap2myo_demo/ik/` | the calibration (`2014001_offsets.npz`), the IK (`2014001_C3_03.npz`), the QA (`qa.json`) and the report (`report.md`) |
| `mocap2myo_demo/kin/` | the joint loads |
| `mocap2myo_demo/viewer/` | the clip |

{doc}`results` explains each file.
