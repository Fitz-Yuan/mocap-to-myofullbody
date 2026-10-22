(video)=
# From video

An ordinary video of one person, from a phone for example, can go in instead of marker data. [GVHMR](https://github.com/zju3dv/GVHMR) (Shen et al. 2024) reconstructs the body in the video as an SMPL mesh in a gravity-aligned world. `mocap2myo video` places 32 virtual markers on that mesh and writes them as an ordinary marker recording. From there it is the usual pipeline: calibration, IK, QA and the exports.

Video gives **joint angles only**, and less accurate ones than markers (see [What to expect](#what-to-expect)). The joint loads are computed but not validated.

## 1. Install GVHMR (once)

GVHMR and the SMPL-X body model it uses have their own licences (research and non-profit use; SMPL-X needs a free registration), so they are not part of this repository. Install them next to it:

1. Follow GVHMR's installation guide (its `docs/INSTALL.md`): a conda environment, its checkpoints, and the SMPL and SMPL-X model files (register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de)). It needs an NVIDIA GPU.
2. Check that its demo runs on one of your videos: `cd GVHMR && python tools/demo/demo.py --video=my_video.mp4`.
3. Note two paths: GVHMR's folder, and the Python of its environment (`conda activate gvhmr`, then `which python`).
4. Install ffmpeg if it is not there (`conda install ffmpeg` or `apt install ffmpeg`).

Problems we met while installing it:

- **Newer NVIDIA GPUs** (Blackwell, RTX 50xx): GVHMR's requirements pin PyTorch 2.3 for CUDA 12.1 and a prebuilt pytorch3d, which do not run there. Install a PyTorch built for CUDA 12.8 instead (remove the torch, torchvision and pytorch3d lines from its `requirements.txt`) and build pytorch3d from source against it.
- **PyTorch 2.6 or newer** refuses GVHMR's checkpoints: set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (`mocap2myo video` sets it itself).
- **The camera estimate can fail on frames where the camera only turns.** If GVHMR stops there, film from a tripod and pass `--static-cam`.

## 2. Record

- One person, the whole body in view all the time, feet included.
- A tripod (then use `--static-cam`), or a phone following the person steadily.
- Level ground; for stairs or slopes, use `--floor as-is`.
- Measure the person's height and weight. A single camera fixes the body's size only roughly; the measured height sets it.

Phone videos are fine as they are: `mocap2myo video` re-encodes each one at a constant 30 frames per second, the way GVHMR expects, and turns portrait videos upright.

## 3. Run

```bash
mocap2myo video walk.mp4 --subject P01 --height 178 --mass 70 --out my_videos \
    --gvhmr ~/GVHMR --gvhmr-python ~/miniconda3/envs/gvhmr/bin/python
mocap2myo check --dataset my_videos/my_videos.toml
mocap2myo run --dataset my_videos/my_videos.toml
```

`M2M_GVHMR` and `M2M_GVHMR_PYTHON` can hold the two paths instead of the options. More videos of the same person: run `mocap2myo video` again with the same `--subject` and `--out`. Another person: another `--subject`. A video GVHMR has already processed: `--result path/to/hmr4d_results.pt`.

For each video it:

1. re-encodes the video at 30 frames per second (upright);
2. runs GVHMR on it (a few minutes per minute of video on a GPU);
3. places the 32 virtual markers on the reconstructed mesh, with one body shape per person (from their first video);
4. scales the whole recording to the measured height;
5. puts the floor at height 0 and removes the reconstruction's slow vertical drift (`--floor flat`, the default);
6. holds the foot that is on the floor in place, because a reconstruction from one camera lets the stance foot slide (`--no-foot-lock` keeps the sliding);
7. writes `raw/<subject>/<subject>_<video>.trc`, a static trial from the same body standing (`<subject>_static.trc`), the height and mass (`anthropometrics.csv`) and the dataset profile.

The virtual markers are those of OpenCap's marker set (Uhlrich et al. 2023), with OpenCapBench's choice of mesh vertex for each: the pelvis, both sides of the knees, ankles, elbows and wrists, the heels, toes and fifth metatarsal heads, the shoulders, C7, T6, T11 and the sternum (`mocap2myo/markersets/smpl_opencap.toml`). The reconstruction does not track the hands or the head, so there are no markers there.

## What to expect

On three phone videos by our lab, filmed with a phone following the person (35, 44 and 31 s):

| Video | QA | Virtual-marker residual (median, mm) |
|---|---|---|
| Walking | PASS | 5.4 |
| Walking and jogging | WARN: standing hip flexion 13° | 5.1 |
| In the lab, the person half out of the frame | FAIL: an arm lost for 0.7 s | 6.4 |

The residual says how well the model fits the reconstructed mesh, **not how close the angles are to the person's**. The angles are only as good as GVHMR's reconstruction: from one camera that is several centimetres per joint, worse for the hands and for anything the camera does not see. A comparison with simultaneous marker capture is planned, on OpenCap's public data (videos and markers recorded together).

The viewer has a [demo](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/?demo=video) of the walking video: the motion only, without the video.
