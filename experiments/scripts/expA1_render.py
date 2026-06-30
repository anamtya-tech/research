#!/usr/bin/env python3
"""EXP-A1 (Track A, ODAS-free) — generate a directional-only scene (no ambient) and render it
headlessly via the simulator's AudioRenderer, producing the .raw + per-source .f32 GT sidecars
+ metadata JSON that gt_dataset_builder consumes.

Honors the experiments.pdf fixed protocol: room 250x250x20, absorption 0.7, max_order 3,
source distances 10-150 m, labels {Elephant, Bear, Frog, Lion, drone_bebop, drone_binary}.

  python expA1_render.py --duration 600 --instances 12 --seed 1 --name exp_a1_no_ambient
  python expA1_render.py --duration 300 --instances 8  --seed 99 --name exp_a1_holdout   # holdout
"""
import os, sys, json, glob, argparse, random, math
import unittest.mock as mock

ROOT = "/Users/abhinav/research"
SOUNDS = f"{ROOT}/backup/sounds"
WORK = f"{ROOT}/experiments/sim"          # local working tree (replaces /home/azureuser)
SCENES = f"{WORK}/scenes"; RENDERS = f"{WORK}/renders"

# experiments.pdf scene labels -> source dir(s). Multiple dirs per class = more clip variety.
AC = f"{ROOT}/backup/audio_cache"
LABEL_DIRS = {
    "Elephant":     [f"{SOUNDS}/jungle_animals/Elephant", f"{AC}/wild_animals/Animal-Soundprepros/Elephant", f"{AC}/elephant_samples_new"],
    "Bear":         [f"{SOUNDS}/jungle_animals/Bear", f"{AC}/wild_animals/Animal-Soundprepros/Bear"],
    "Frog":         [f"{SOUNDS}/jungle_animals/Frog", f"{AC}/wild_animals/Animal-Soundprepros/Frog"],
    "Lion":         [f"{SOUNDS}/jungle_animals/Lion", f"{AC}/wild_animals/Animal-Soundprepros/Aslan"],
    "drone_bebop":  [f"{SOUNDS}/machine_sounds/drone_multi/drone_bebop"],
    "drone_binary": [f"{SOUNDS}/machine_sounds/drone_binary"],
}
# fixed protocol
ROOM = (250.0, 250.0, 20.0); ABSORPTION = 0.7; MAX_ORDER = 3
DIST_MIN, DIST_MAX = 10.0, 150.0
import soundfile as sf

def read_label_meta(label):
    dirs = LABEL_DIRS[label]
    if isinstance(dirs, str): dirs = [dirs]
    spl, stype, wavs = None, "directional", []
    for d in dirs:
        lt = os.path.join(d, "label.txt")
        if spl is None and os.path.exists(lt):
            lines = [l.strip() for l in open(lt)]
            if len(lines) >= 2: stype = lines[1] or stype
            if len(lines) >= 3:
                try: spl = float(lines[2])
                except ValueError: pass
        wavs += sorted(glob.glob(os.path.join(d, "*.wav")) + glob.glob(os.path.join(d, "*.mp3")))
    return wavs, spl, stype

def clip_dur(path):
    try:
        info = sf.info(path); return info.frames / info.samplerate
    except Exception:
        return 2.0

def make_scene(name, duration, instances, seed, capture=None, cap_volume=1.0, cap_offset=0.0, only=None):
    rng = random.Random(seed)
    sources = []
    labels = only if only else list(LABEL_DIRS)
    for label in labels:
        wavs, spl, _ = read_label_meta(label)
        if not wavs:
            print(f"  WARN: no wavs for {label}"); continue
        for _ in range(instances):
            wav = rng.choice(wavs)
            dur = min(clip_dur(wav), 8.0)
            az = math.radians(rng.uniform(-180, 180))
            dist = rng.uniform(DIST_MIN, DIST_MAX)
            start = rng.uniform(0, max(0.1, duration - dur))
            src = dict(label=label, wav_path=wav,
                       x=round(dist*math.cos(az), 2), y=round(dist*math.sin(az), 2), z=0.0,
                       start_time=round(start, 2), end_time=round(start + dur, 2),
                       repeat=False, volume=1.0)
            if spl is not None:
                src["spl_db_1m"] = spl; src["spl_defaulted"] = False
            else:
                src["spl_db_1m"] = 80.0; src["spl_defaulted"] = True
            sources.append(src)
    rng.shuffle(sources)
    scene = dict(name=name, duration=float(duration), max_radius=DIST_MAX, max_height=2.0,
                 min_height=0.0, directional_sources=sources, ambient_sources=[],
                 ambient_mode="synthetic", version="1.0")
    if capture:
        scene["ambient_mode"] = "capture"
        scene["ambient_capture"] = dict(path=capture, volume=float(cap_volume),
                                        start_offset=float(cap_offset))
    return scene

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="exp_a1_no_ambient")
    ap.add_argument("--capture", default=None, help="real .raw capture for ambient_mode=capture")
    ap.add_argument("--cap-volume", type=float, default=1.0)
    ap.add_argument("--cap-offset", type=float, default=0.0)
    ap.add_argument("--only", default=None, help="comma-separated subset of classes to render")
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None
    os.makedirs(SCENES, exist_ok=True); os.makedirs(RENDERS, exist_ok=True)
    scene = make_scene(args.name, args.duration, args.instances, args.seed,
                       capture=args.capture, cap_volume=args.cap_volume, cap_offset=args.cap_offset, only=only)
    scene_path = f"{SCENES}/{args.name}.json"
    json.dump(scene, open(scene_path, "w"), indent=2)
    print(f"scene: {len(scene['directional_sources'])} directional sources, "
          f"{args.duration}s, 0 ambient -> {scene_path}")

    # headless: stub streamlit, then drive the real renderer
    sys.modules["streamlit"] = mock.MagicMock()
    sys.path.insert(0, f"{ROOT}/simulator")
    from renderer import AudioRenderer
    r = AudioRenderer(SCENES, RENDERS)
    out = r._render_scene(scene, ROOM[0], ROOM[1], ROOM[2], ABSORPTION, MAX_ORDER,
                          add_noise=False, noise_level=0.0)
    meta_path = str(out).replace(".raw", ".json")
    meta = json.load(open(meta_path))
    print(f"rendered: {out}")
    print(f"  duration={meta['duration']}s  channels={meta['n_channels']}  "
          f"sidecars={len(meta['source_sidecars'])}  ambient_sidecar={'ambient_sidecar_path' in meta}")
    print(f"  metadata: {meta_path}")

if __name__ == "__main__":
    main()
