"""ResNet14/CIFAR-10 teaching harness; see ../GUIDE.md for experiment scope.

Each invocation evaluates one condition. Synthetic --smoke results are explicitly
labelled and must never be used as CIFAR-10 accuracy measurements.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision import datasets, transforms
from applications.dnn.torch.cifar10_resnet.build_resnet_cifar10 import ResNet_cifar10
from applications.dnn.dnn_inference_params import dnn_inference_params
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, analog_modules

CASES = ['fp', 'ideal', 'w8', 'w8a8', 'q8', 'program', 'read', 'write-read',
         'retention', 'adc', 'rp', 'bl', 'calibrate']
EXAMPLE = ROOT / 'applications/dnn/torch/cifar10_resnet'


def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        import cupy as cp
        cp.random.seed(seed)
        torch.cuda.manual_seed_all(seed)


def parameters(args, n_layers):
    ranges = np.load(EXAMPLE / 'calibrated_config/input_limits_ResNet14.npy')
    adc_path = args.adc_limits or EXAMPLE / 'calibrated_config/adc_limits_ResNet14_balanced.npy'
    adcs = np.load(adc_path)
    if ranges.shape != (n_layers, 2) or adcs.shape != (n_layers, 2):
        raise ValueError('Calibration tables must have one [min,max] per converted layer')
    cfg = dict(ideal=False, core_style='BALANCED', balanced_style='ONE_SIDED',
               Nslices=1, weight_bits=8, weight_percentile=100, digital_bias=True,
               Rmin=62500., Rmax=6.25e11, Vread=0.1, infinite_on_off_ratio=False,
               error_model='none', noise_model='none', drift_model='none', t_drift=0,
               NrowsMax=args.rows, NcolsMax=None, Rp_row=0., Rp_col=0.,
               interleaved_posneg=False, subtract_current_in_xbar=True,
               input_bits=8, input_bitslicing=True, input_slice_size=1,
               current_from_input=False, selected_rows='top', adc_bits=8,
               adc_range_option='CALIBRATED', adc_type='generic', adc_per_ibit=False,
               useGPU=args.gpu, lumped_read_noise=False)
    if args.case in ['ideal', 'w8']:
        cfg.update(input_bits=0, input_bitslicing=False, current_from_input=True, adc_bits=0)
    if args.case == 'ideal':
        cfg['weight_bits'] = 0
    if args.case in ['w8a8', 'calibrate']:
        cfg['adc_bits'] = 0
    if args.case in ['program', 'write-read', 'retention']:
        cfg['error_model'] = 'SONOS'
    if args.case in ['read', 'write-read', 'retention']:
        if args.noise_model == 'paper':
            cfg.update(noise_model='generic', alpha_noise=0.0087, proportional_noise=True)
        else:
            cfg['noise_model'] = 'SONOS'
    if args.case == 'retention':
        cfg.update(drift_model='SONOS', t_drift=args.days)
    if args.case == 'adc':
        cfg['adc_bits'] = args.bits
    if args.case == 'rp':
        cfg['Rp_col'] = args.rp
    params = []
    for k in range(n_layers):
        p = dnn_inference_params(**cfg, positiveInputsOnly=k > 0,
                                 input_range=ranges[k].copy(), adc_range=adcs[k].copy())
        if cfg['weight_bits'] == 0:
            p.xbar.device.cell_bits = 0
        if args.case == 'bl':
            p.simulation.disable_fast_balanced = True
        params.append(p)
    return params, cfg, Path(adc_path)


def attach_bl_collectors(model, gpu):
    """Reconstruct nonnegative LSB branch currents before subtraction, at Rp=0.

    This is analytic current bookkeeping from the actual programmed G and DAC
    bit vectors. For signed inputs use complementary row pairs; do not mistake
    a signed CrossSim branch result for a physical nonnegative BL current.
    """
    xp = np
    if gpu:
        import cupy as xp
    stats = {}
    for li, layer in enumerate(analog_modules(model)):
        for ri, row in enumerate(layer.core.cores):
            for ci, core in enumerate(row):
                key = f'layer{li}_partition{ri}_{ci}'
                stats[key] = dict(logical_rows=int(core.core_pos.matrix.shape[1]),
                                  signed=bool(core.dac_params.mvm.signed),
                                  sum_pos_norm=0., sum_neg_norm=0., samples=0, calls=0)
                original = core.run_xbar_operation

                def wrapped(op, vector, *, core=core, original=original, key=key):
                    if op == 'mvm':
                        bit = core.dac.mvm.convert_sliced(vector)[0]
                        bp, bn = xp.maximum(bit, 0), xp.maximum(-bit, 0)
                        gp, gn = core.core_pos.matrix, core.core_neg.matrix
                        ip = gp @ bp + gn @ bn
                        im = gn @ bp + gp @ bn
                        s = stats[key]
                        s['sum_pos_norm'] += float(ip.sum())
                        s['sum_neg_norm'] += float(im.sum())
                        s['samples'] += int(ip.size)
                        s['calls'] += 1
                    return original(op, vector)

                core.run_xbar_operation = wrapped
    return stats


def summarize_bl(stats):
    groups = defaultdict(list)
    arrays = []
    for key, s in stats.items():
        if not s['samples']:
            raise RuntimeError(f'No current samples collected for {key}')
        row = dict(array=key, **s)
        row['mean_pos_uA'] = 1.6 * s['sum_pos_norm'] / s['samples']
        row['mean_neg_uA'] = 1.6 * s['sum_neg_norm'] / s['samples']
        row['physical_rows_if_signed_pair'] = s['logical_rows'] * (2 if s['signed'] else 1)
        arrays.append(row)
        groups[s['logical_rows']].append(row)
    result = []
    for nrows, rows in sorted(groups.items()):
        count = sum(r['samples'] for r in rows)
        result.append(dict(rows=nrows, arrays=len(rows),
                           mean_pos_uA=float(np.mean([r['mean_pos_uA'] for r in rows])),
                           mean_neg_uA=float(np.mean([r['mean_neg_uA'] for r in rows])),
                           sample_weighted_pos_uA=1.6*sum(r['sum_pos_norm'] for r in rows)/count,
                           sample_weighted_neg_uA=1.6*sum(r['sum_neg_norm'] for r in rows)/count))
    return dict(by_rows=result, per_array=arrays, bit='LSB', weighting='equal physical arrays, then mean over BLs/windows/images within each array',
                signed_input='logical row count shown; signed input row pairs reconstructed analytically',
                interpretation='Rp=0 analytic BL current reconstruction; not a transistor circuit measurement')


def attach_adc_collectors(model):
    """Capture final SIR sum at the ADC call, not raw per-input-bit analytics."""
    samples = [[] for _ in analog_modules(model)]
    for li, layer in enumerate(analog_modules(model)):
        for row in layer.core.cores:
            for core in row:
                adc = core.adc.mvm
                original = adc.convert

                def wrapped(value, *, li=li, original=original):
                    array = value.get() if hasattr(value, 'get') else np.asarray(value)
                    samples[li].append(array.astype(np.float32).reshape(-1).copy())
                    return original(value)

                adc.convert = wrapped
    return samples


def loader(args):
    if args.smoke:
        generator = torch.Generator().manual_seed(args.data_seed)
        data = torch.randn(args.n, 3, 32, 32, generator=generator)
        labels = torch.zeros(args.n, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(data, labels)
        ids = list(range(args.n))
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ds = datasets.CIFAR10(str(args.data_root), train=args.case == 'calibrate',
                             transform=transform, download=args.download)
        if args.n > len(ds):
            raise ValueError('Requested more images than the dataset contains')
        ids = np.random.default_rng(args.data_seed).permutation(len(ds))[:args.n].tolist()
        ds = torch.utils.data.Subset(ds, ids)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0), ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--nruns', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--data-seed', type=int, default=314159)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--noise-model', choices=['paper', 'native'], default='paper')
    parser.add_argument('--rows', type=int, default=1152)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--rp', type=float, default=0.)
    parser.add_argument('--days', type=float, default=1.)
    parser.add_argument('--adc-limits', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT/'scratch/sim-work/data/cifar10')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; choose a new path to preserve previous measurements')
    if args.n <= 0 or args.nruns <= 0 or args.rows <= 0 or args.batch_size <= 0:
        parser.error('n, nruns, rows and batch-size must be positive')
    if args.rp < 0 or not 0 <= args.days <= 5 or not 2 <= args.bits <= 10:
        parser.error('Use rp>=0, days in [0,5], bits in [2,10]')
    if args.gpu and not torch.cuda.is_available():
        parser.error('GPU requested but torch.cuda.is_available() is false')
    if args.case in ['bl', 'calibrate'] and args.nruns != 1:
        parser.error('bl/calibrate use nruns=1; repeated accuracy trials are different experiments')
    if args.case == 'calibrate' and args.adc_limits:
        parser.error('Calibration creates a new range file; do not specify --adc-limits')
    device = 'cuda:0' if args.gpu else 'cpu'
    torch.set_num_threads(4)
    data_loader, indices = loader(args)
    checkpoint = EXAMPLE/'models/resnet14_cifar10.pth'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runs = []
    metadata = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    synthetic=args.smoke, created_utc=datetime.now(timezone.utc).isoformat(),
                    torch=torch.__version__, checkpoint_sha256=checkpoint_hash, indices=indices,
                    sonos_fit='Current repository IEDM-2022 model; not the original TCAS-I fit',
                    source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    for run in range(args.nruns):
        seed = args.seed + run
        seed_all(seed, args.gpu)
        model = ResNet_cifar10(2).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        n_layers = len(convertible_modules(model))
        metadata['converted_layers'] = n_layers
        if args.case != 'fp':
            params, cfg, adc_path = parameters(args, n_layers)
            model = from_torch(model, params, fuse_batchnorm=True, bias_rows=0).eval()
            metadata['base_params'] = cfg
            metadata['adc_limits_path'] = str(adc_path)
            metadata['adc_limits_sha256'] = hashlib.sha256(adc_path.read_bytes()).hexdigest()
            metadata['input_limits_sha256'] = hashlib.sha256((EXAMPLE/'calibrated_config/input_limits_ResNet14.npy').read_bytes()).hexdigest()
        bl_stats = attach_bl_collectors(model, args.gpu) if args.case == 'bl' else None
        adc_samples = attach_adc_collectors(model) if args.case == 'calibrate' else None
        correct1 = correct5 = seen = 0
        start = time.perf_counter()
        with torch.no_grad():
            for x,y in data_loader:
                x,y = x.to(device),y.to(device)
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Non-finite logits; do not record this as classification accuracy')
                best = logits.topk(5, dim=1).indices
                correct1 += int((best[:,0] == y).sum())
                correct5 += int((best == y[:,None]).any(1).sum())
                seen += y.numel()
        if args.gpu:
            torch.cuda.synchronize()
        result = dict(run=run, seed=seed, n=seen, correct1=correct1, correct5=correct5,
                      top1_pct=100*correct1/seen, top5_pct=100*correct5/seen,
                      seconds=time.perf_counter()-start)
        runs.append(result)
        print(('SMOKE ONLY ' if args.smoke else '')+json.dumps(result),flush=True)
        if bl_stats is not None:
            metadata['bl'] = summarize_bl(bl_stats)
        if adc_samples is not None:
            from applications.dnn.calibration import calibrate_adc_limits
            collected = [np.concatenate(values)[None,:] for values in adc_samples]
            limits = calibrate_adc_limits(analog_modules(model), collected, Nbits=12, norm_ord=1.)
            target = args.out.with_suffix('.adc.npy')
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            np.save(target,limits)
            metadata['calibration'] = dict(path=str(target), Nbits=12, norm_ord=1.,
                                          samples_per_layer=[int(v.size) for v in collected],
                                          signal='final accumulated pre-ADC SIR signal')
        del model
        if args.gpu:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
    metadata['runs'] = runs
    for metric in ['top1_pct','top5_pct']:
        values = [r[metric] for r in runs]
        metadata[metric] = dict(mean=float(np.mean(values)),
                               sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',args.out,flush=True)


if __name__ == '__main__':
    main()
