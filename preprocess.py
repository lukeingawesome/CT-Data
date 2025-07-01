# preprocess_ct_to_npz.py
import os, json, argparse, multiprocessing as mp, numpy as np, nibabel as nib
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, CastToTyped
)

# ---------- constants --------------------------------------------------------
TARGET_SPACING   = (1.5, 1.5, 3.0)          # (x, y, z) mm
TARGET_SHAPE     = (224, 224, 160)          # (H, W, D) after crop
HU_WINDOW        = (-1000, 1500)            # clip & scale to [0,1]
NPZ_DTYPE        = np.float16               # 2 bytes / voxel
N_PROC           = max(mp.cpu_count() - 2, 4)
# -----------------------------------------------------------------------------

# ---------- constants -------------------------------------------------------
TARGET_SPACING   = (1.25, 1.25, 2.0)        # (x, y, z) mm – less information loss
TARGET_SHAPE     = (256, 256, 192)          # (H, W, D) after tight crop + pad
HU_WINDOW        = (-1000, 1500)            # clip & scale to [0,1]
NPZ_DTYPE        = np.float16               # storage only!
N_PROC           = max(mp.cpu_count() - 2, 4)
# ---------------------------------------------------------------------------

offline_tx = Compose([
    LoadImaged(keys="image"),                               # ITK loader
    EnsureChannelFirstd(keys="image"),                      # (C, Z, Y, X)
    Orientationd(keys="image", axcodes="RAS"),              # world‑standard
    Spacingd(keys="image", pixdim=TARGET_SPACING,
             mode="trilinear", align_corners=True),
    ScaleIntensityRanged(keys="image",
        a_min=HU_WINDOW[0], a_max=HU_WINDOW[1],
        b_min=0.0, b_max=1.0, clip=True),
    SpatialPadd(keys="image", spatial_size=(*TARGET_SHAPE,)),
    CenterSpatialCropd(keys="image", roi_size=(*TARGET_SHAPE,)),
    CastToTyped(keys="image", dtype=NPZ_DTYPE),             # float16
])

def process_one(path_outdir_pair):
    path, out_dir = path_outdir_pair
    case_id = Path(path).stem.replace(".nii", "")
    out_file = out_dir / f"{case_id}.npz"
    
    # Load original image to get metadata before preprocessing
    original_img = nib.load(path)
    original_shape = original_img.shape
    
    if out_file.exists():        # skip if already done
        return {
            'img_path': str(path),
            'case_id': case_id,
            'original_shape': original_shape,
            'processed_file': str(out_file),
            'status': 'already_exists'
        }
    
    try:
        data_dict = offline_tx({"image": path})
        vol = data_dict["image"]          # shape (1, Z, Y, X)
        # Move to (C, H, W, D) so PyTorch works with .permute in training if needed
        vol = vol.astype(NPZ_DTYPE)       # already float16
        np.savez_compressed(out_file, image=vol)
        
        return {
            'img_path': str(path),
            'case_id': case_id,
            'original_shape': original_shape,
            'processed_file': str(out_file),
            'status': 'processed'
        }
    except Exception as e:
        return {
            'img_path': str(path),
            'case_id': case_id,
            'original_shape': original_shape,
            'processed_file': None,
            'status': f'error: {str(e)}'
        }

def main(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    nii_paths = [p for p in src_dir.rglob("*.nii.gz")]
    print(f"Found {len(nii_paths)} .nii.gz files")

    print(f"Starting preprocessing with {N_PROC} processes...")
    with mp.Pool(N_PROC) as pool:
        results = list(tqdm(
            pool.imap(process_one, [(p, dst_dir) for p in nii_paths]),
            total=len(nii_paths), 
            desc="Processing CT scans",
            unit="files",
            dynamic_ncols=True,
            ncols=100
        ))
    
    # Filter out None results and create dataframe
    metadata_list = [r for r in results if r is not None]
    
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        
        # Add original shape components as separate columns for easier analysis
        df['original_height'] = df['original_shape'].apply(lambda x: x[0])
        df['original_width'] = df['original_shape'].apply(lambda x: x[1])
        df['original_depth'] = df['original_shape'].apply(lambda x: x[2])
        
        # Save metadata as CSV
        metadata_file = dst_dir / "preprocessing_metadata.csv"
        df.to_csv(metadata_file, index=False)
        print(f"Saved metadata for {len(df)} files to {metadata_file}")
        
        # Print summary statistics
        print("\nProcessing Summary:")
        print(df['status'].value_counts())
        
        if 'processed' in df['status'].values:
            processed_df = df[df['status'] == 'processed']
            print(f"\nOriginal shape statistics for processed files:")
            print(f"Height: min={processed_df['original_height'].min()}, max={processed_df['original_height'].max()}, mean={processed_df['original_height'].mean():.1f}")
            print(f"Width: min={processed_df['original_width'].min()}, max={processed_df['original_width'].max()}, mean={processed_df['original_width'].mean():.1f}")
            print(f"Depth: min={processed_df['original_depth'].min()}, max={processed_df['original_depth'].max()}, mean={processed_df['original_depth'].mean():.1f}")
    
    else:
        print("No metadata collected.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/bilab7/data/train", help="folder with original .nii.gz")
    ap.add_argument("--dst", default="/home/bilab7/data/train_preprocessed3", help="output folder for .npz")
    args = ap.parse_args()
    main(args.src, args.dst)


import numpy as np
import torch
from pathlib import Path
import monai
from typing import List, Dict, Any

class NPZDataset(monai.data.Dataset):
    """
    Returns:
        dict(
            image = FloatTensor (C=2, D, H, W) –‑ **float32 for training**,
            meta  = { 'id': <str> }
        )
    """
    def __init__(self, files: List[Path]):
        self.files = files
        super().__init__(data=files)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = self.files[idx]
        arr = np.load(fname)["image"]       # (C, D, H, W) float16
        arr = torch.from_numpy(arr).float() # cast to float32 for gradients
        return {"image": arr, "meta": {"id": fname.stem}}

def get_loader(npz_root: str,
               batchsize: int = 2,
               shuffle: bool = True,
               num_workers: int = 8):
    files = sorted(Path(npz_root).glob("*.npz"))
    ds    = NPZDataset(files)
    return monai.data.DataLoader(ds,
                                 batch_size=batchsize,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 persistent_workers=num_workers > 0)