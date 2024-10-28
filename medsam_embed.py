#%%
from segment_anything import sam_model_registry
import numpy as np
from tqdm import tqdm
import torch
import shutil
import nibabel as nib
import os, sys
import zipfile
import argparse
#%%
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--dataroot', type=str, default='/u/project/sgss/UKBB/imaging/bulk/20253', help='Data root directory')
parser.add_argument('--encoder', type=str, default='SAM', help='Encoder type')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--manifest', type=str, required=True, help='Manifest file path')
parser.add_argument('--start', type=int, required=True, help='Start index')
parser.add_argument('--many', type=int, required=True, help='Number of files to process')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--saveto', type=str, required=True, help='Save directory')
parser.add_argument('--all_slices', action='store_true', default=False)
parser.add_argument('--save_sam', action='store_true', default=False)

args = parser.parse_args()

sam_checkpoints = dict(
    MedSAM='work_dir/MedSAM/medsam_vit_b.pth',
    SAM='work_dir/MedSAM/sam_vit_b_01ec64.pth',
)

with open(args.manifest) as fl:
    fls = [ln.strip() for ln in fl if ln]
fbatch = fls[args.start:args.start+args.many]
#%%
medsam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoints[args.encoder])
medsam_model = medsam_model.to(args.device)
medsam_model.eval()
#%%

def crop_pad_matrix(mat, size=256):
    h, w = mat.shape
    if h < size:
        pad_h = (size - h) // 2
        pad_h = (pad_h, pad_h + (size - h) % 2)
        mat = np.pad(mat, (pad_h, (0, 0)), mode='constant')
    else:
        mat = mat[(h - size) // 2:(h + size) // 2, :]
    if w < size:
        pad_w = (size - w) // 2
        pad_w = (pad_w, pad_w + (size - w) % 2)
        mat = np.pad(mat, ((0, 0), pad_w), mode='constant')
    else:
        mat = mat[:, (w - size) // 2:(w + size) // 2]
    return mat


pbar = tqdm(fbatch)
for fname in pbar:
    pid = fname.split('/')[-1].split('_')[0]

    zipname = f'{args.dataroot}/{fname}'


    temp_folder = f'temp_{args.saveto}/{pid}'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)
    with zipfile.ZipFile(zipname, 'r') as zip_ref:
        if 'T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz' in zip_ref.namelist():
            zip_ref.extract('T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz', path=temp_folder)
            file_path = f'{temp_folder}/T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz'
        else:
            continue

    vol = nib.load(file_path).get_fdata()
    minval, maxval = vol.min(), vol.max()
    shutil.rmtree(temp_folder)

    pbar.set_postfix(dict(
        pid=pid, min=minval, max=maxval, sh=vol.shape
    ))

    hclip = 1024+256
    vol[vol < 0] = 0
    vol[vol > hclip] = hclip
    vol = vol.astype(float)
    vol /= hclip

    # collect slices (in axes order)
    slices_byxyz = []

    slices = []
    for i in (range(5, vol.shape[0]-5) if args.all_slices else range(5, vol.shape[0], 10)):
        slices += [vol[i]]
    nx = len(slices)
    slices_byxyz += slices

    slices = []
    for i in (range(5, vol.shape[1]-5) if args.all_slices else range(5, vol.shape[1], 10)):
        slices += [vol[:, i]]
    ny = len(slices)
    slices_byxyz += slices

    slices = []
    for i in (range(5, vol.shape[2]-5) if args.all_slices else range(5, vol.shape[2], 10)):
        slices += [vol[:, :, i]]
    nz = len(slices)
    slices_byxyz += slices

    # projs_byxyz = []
    # for side in slices_byxyz:
    imgs = [crop_pad_matrix(img) for img in slices_byxyz]
    embs = []
    for i in range(0, len(imgs), args.batch_size):
        imgbatch = np.array(imgs[i:i+args.batch_size]).astype(np.float32)
        imgbatch = torch.from_numpy(imgbatch[:, None]).repeat(1, 3, 1, 1).to(args.device)
        with torch.no_grad():
            embs += [e for e in medsam_model.image_encoder(imgbatch).detach().cpu().numpy()]

    if args.save_sam:
        np.savez_compressed(f'{args.saveto}/{pid}.npz', np.array(embs).astype(np.float32))
        continue

    for projector in ['artifacts/proj_normal_k10.npy', 'artifacts/proj_normal_k100.npy', None]:
        if projector is not None:
            projmat = np.load(projector)
            projname = projector.split('/')[-1].split('.')[0]
        else:
            projmat = np.eye(256)
            projname = 'proj_identity'

        if not os.path.exists(f'{args.saveto}/{projname}'):
            os.makedirs(f'{args.saveto}/{projname}')

        # embs: slices (~16) x 256 x 16 x 16
        assert len(projmat) == len(embs[0])
        proj_embs = np.stack([projmat.T @ e.reshape(projmat.shape[0], -1) for e in embs])

        # proj_embs: slices (16 + 16 + 16) x K x 256
        # proj_embs_sum: slices 3 x K x 16 x 16
        assert nx+ny+nz == len(proj_embs)
        byside = [s for s in np.split(proj_embs, [nx, nx+ny, nx+ny+nz], axis=0) if len(s)]
        assert len(byside) == 3
        byside = [side.sum(0) for side in byside]
        proj_embs_sum = np.concatenate(byside)

        # proj_embs_sum: slices 3K x 256 ~ 7680 for K=10
        proj_embs_sum_flat = proj_embs_sum.reshape(-1)

        np.savez_compressed(f'{args.saveto}/{projname}/{pid}.npz', proj_embs_sum_flat)

    # assert False
#%%