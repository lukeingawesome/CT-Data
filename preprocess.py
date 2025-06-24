# preprocess_ct_to_npz.py
# ---------------------------------------------------------------------------
# Offline conversion of chest CT (NIfTI) to 2‑channel, fixed‑size NPZ volumes
# ---------------------------------------------------------------------------
import os, json, argparse, multiprocessing as mp, hashlib
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CopyItemsd, ScaleIntensityRanged, ConcatItemsd,
)

# ---------- constants -------------------------------------------------------
TARGET_SPACING   = (1.25, 1.25, 2.0)        # (x, y, z) mm – less information loss
TARGET_SHAPE     = (256, 256, 192)          # (H, W, D) after tight crop + pad
HU_WINDOWS       = [(-1000,  400),          # soft‑tissue / lung parenchyma
                    (-1000, 2000)]          # bone / high density
NPZ_DTYPE        = np.float16               # storage only!
N_PROC           = max(mp.cpu_count() - 2, 4)
LUNG_MARGIN      = 16                       # voxels to keep around lung bbox
# ---------------------------------------------------------------------------

mp.set_start_method("forkserver", force=True)

# --- MONAI transforms that are the same for both channels -------------------
base_tx = Compose([
    LoadImaged(keys="image"),                    # ITK loader
    EnsureChannelFirstd(keys="image"),           # (1, Z, Y, X)
    Orientationd(keys="image", axcodes="RAS"),   # standard orientation
    Spacingd(keys="image",
             pixdim=TARGET_SPACING,
             mode="trilinear",
             align_corners=False),
])

# window 1 and window 2 pipelines -> concat along channel dim (0)
win_tx = Compose([
    CopyItemsd(keys="image", times=1, names=["image2"]),
    ScaleIntensityRanged(keys="image",
                         a_min=HU_WINDOWS[0][0], a_max=HU_WINDOWS[0][1],
                         b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys="image2",
                         a_min=HU_WINDOWS[1][0], a_max=HU_WINDOWS[1][1],
                         b_min=0.0, b_max=1.0, clip=True),
    ConcatItemsd(keys=["image", "image2"], name="image", dim=0),
])

def tight_crop(vol: np.ndarray, margin: int = 16) -> np.ndarray:
    """
    Roughly isolate the lung ROI by cropping to the bounding box of non‑air voxels
    and add a fixed margin. Works on a (C, Z, Y, X) float array in [0,1].
    """
    # Non‑air voxels: intensity > 0.05 of windowed scale
    mask = (vol > 0.05).any(0)   # (Z, Y, X)
    coords = np.array(np.where(mask))
    if coords.size == 0:                       # fallback – all air
        return vol
    zmin, ymin, xmin = coords.min(1) - margin
    zmax, ymax, xmax = coords.max(1) + margin + 1
    zmin, ymin, xmin = np.maximum((zmin, ymin, xmin), 0)
    zmax, ymax, xmax = np.minimum((zmax, ymax, xmax), vol.shape[1:])
    return vol[:, zmin:zmax, ymin:ymax, xmin:xmax]

def process_one(path_outdir_pair):
    path, out_dir = path_outdir_pair
    case_id = Path(path).stem.replace(".nii", "")

    # disambiguate identically named files from different folders
    file_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]
    out_file  = out_dir / f"{case_id}_{file_hash}.npz"
    meta_file = out_dir / f"{case_id}_{file_hash}.json"

    # load once for metadata
    original_img    = nib.load(path)
    original_shape  = original_img.shape
    original_affine = original_img.affine.tolist()          # JSON serialisable

    if out_file.exists():
        return dict(img_path=str(path),
                    case_id=case_id,
                    original_shape=original_shape,
                    processed_file=str(out_file),
                    status='already_exists')

    try:
        data = base_tx({"image": path})
        data = win_tx(data)
        vol  = data["image"].astype(np.float32)             # keep precision for crop
        vol  = tight_crop(vol, margin=LUNG_MARGIN)

        # Pad to TARGET_SHAPE (H, W, D) -> vol is (C, Z, Y, X)
        pad_H = max(0, TARGET_SHAPE[0] - vol.shape[2])
        pad_W = max(0, TARGET_SHAPE[1] - vol.shape[3])
        pad_D = max(0, TARGET_SHAPE[2] - vol.shape[1])
        pad = [(0, 0),
               (pad_D//2, pad_D - pad_D//2),
               (pad_H//2, pad_H - pad_H//2),
               (pad_W//2, pad_W - pad_W//2)]
        vol = np.pad(vol, pad, mode="constant", constant_values=0.)
        # Centre crop if still larger
        vol = vol[:,
                  vol.shape[1]//2 - TARGET_SHAPE[2]//2 : vol.shape[1]//2 + (TARGET_SHAPE[2]+1)//2,
                  vol.shape[2]//2 - TARGET_SHAPE[0]//2 : vol.shape[2]//2 + (TARGET_SHAPE[0]+1)//2,
                  vol.shape[3]//2 - TARGET_SHAPE[1]//2 : vol.shape[3]//2 + (TARGET_SHAPE[1]+1)//2]

        vol = vol.astype(NPZ_DTYPE)                         # 2‑byte storage
        np.savez_compressed(out_file, image=vol)

        # side‑car JSON – all the provenance you may need later
        meta = dict(
            source_file=str(path),
            case_id=case_id,
            file_hash=file_hash,
            original_shape=original_shape,
            target_shape=vol.shape,
            original_affine=original_affine,
            target_spacing=TARGET_SPACING,
            windows=HU_WINDOWS,
        )
        meta_file.write_text(json.dumps(meta, indent=2))

        return dict(img_path=str(path),
                    case_id=case_id,
                    original_shape=original_shape,
                    processed_file=str(out_file),
                    status='processed')
    except Exception as e:
        return dict(img_path=str(path),
                    case_id=case_id,
                    original_shape=original_shape,
                    processed_file=None,
                    status=f'error: {e}')

def main(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    nii_paths = [p for p in src_dir.rglob("*.nii.gz")]
    # -----------------------------------------------------------------------
    ids = [p.stem.replace(".nii", "") for p in nii_paths]
    assert len(set(ids)) == len(ids), "Duplicate case IDs detected – abort."
    # -----------------------------------------------------------------------

    print(f"Found {len(nii_paths)} .nii.gz files")
    print(f"Starting preprocessing with {N_PROC} processes…")

    with mp.Pool(N_PROC) as pool:
        results = list(tqdm(pool.imap(process_one,
                                      [(p, dst_dir) for p in nii_paths]),
                            total=len(nii_paths),
                            desc="Processing CT scans",
                            unit="files",
                            dynamic_ncols=True,
                            ncols=100))

    df = pd.DataFrame(results)
    # explode shape into columns
    df[['h', 'w', 'd']] = pd.DataFrame(df.original_shape.tolist(), index=df.index)
    df.to_csv(dst_dir / "preprocessing_metadata.csv", index=False)
    print(df.status.value_counts())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/ct_train",
                    help="folder with original *.nii.gz")
    ap.add_argument("--dst", default="/data/train_preprocessed",
                    help="output folder for *.npz + *.json")
    args = ap.parse_args()
    main(args.src, args.dst)
