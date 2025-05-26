import time
import os
import shutil
import zipfile

from tqdm import tqdm
import numpy as np
import zarr

# import napari
# import dask.array as da
# import nibabel as nib
# import pandas as pd

import pydmr
import dbdicom as db
#import wezel
#from wezel.plugins import pyvista, transform, segment
import vreg
import mdreg
import miblab


# import stages
# from compute import io

DATAPATH = "C:\\Users\\md1spsx\\Documents\\Data\\TRISTAN_TWO_COMPOUND"
RESULTSPATH = "C:\\Users\\md1spsx\\Documents\\Results"

SCANS = 'POLARIS_MEDCIC'
T2W = 't2_haste_cor_mbh'
MOLLI = 'T1Map_LL_tra_mbh'
DCE = '3D_DISCO_Dyn_cor_fb'
ROI_SUBSET = ['liver', 'aorta', 'spleen', 'kidney_right', 'kidney_left', 
             'inferior_vena_cava', 'portal_vein_and_splenic_vein']
ROI_MOLLI = ['liver', 'aorta', 'spleen', 'inferior_vena_cava']

TMPPATH = os.path.join(RESULTSPATH, 'tmp')
if not os.path.exists(TMPPATH):
    os.makedirs(TMPPATH)


# helper functions

# def display(dicom_folder):
#     proj = 'TRISTAN'
#     wzl = wezel.app(project=proj)
#     wzl.add_menu(proj)
#     wzl.add_action(pyvista.action_show_mask_surface, proj)
#     wzl.add_action(pyvista.action_show_mask_surfaces, proj)
#     wzl.add_action(pyvista.action_show_mask_surfaces_with_reference, proj)
#     wzl.add_menu(segment.menu)
#     wzl.add_menu(transform.menu)
#     wzl.add_menu(wezel.menubar.about.menu) 
#     wzl.open(dicom_folder).use()
#     #wzl.open(dicom_folder).display('DISCO_mean').use()


def folder(scan):
    visit = scan[:-5]
    scan_nr = scan[-5:]
    results = os.path.join(RESULTSPATH, visit, scan_nr)
    dmrpath = os.path.join(results, 'dmr')
    if not os.path.exists(dmrpath):
        os.makedirs(dmrpath)
    return results

def visit_results(dataset):
    dicom_visit = os.path.join(RESULTSPATH, dataset, 'DICOM')
    if not os.path.exists(dicom_visit):
        os.makedirs(dicom_visit)    
    return os.path.join(RESULTSPATH, dataset)


def _build_visit_dicom(dataset): #real
    subj = dataset[:9]
    path = visit_results(dataset)
    result = os.path.join(path, 'DICOM')
    for n in [1,2]:
        scan = os.path.join(path, f'Scan{n}', dataset+f'Scan{n}')
        study = [scan, db.patients(scan)[0], 'MocoMaps']
        for dce in db.series(study, name=DCE):
            db.copy(dce, [result, subj, f'MocoScan{n}'])

def build_visit_dicom(dataset): # for debugging
    subj = dataset[:9]
    path = visit_results(dataset)
    output = [os.path.join(path, 'DICOM'), subj]
    for n in [1,2]:
        scan = os.path.join(path, f'Scan{n}', dataset+f'Scan{n}')
        S0 = db.series(scan, name='S0')[0]
        db.copy(S0, output + [f'TEST{n}'])


def dummy(scan):

    path = os.path.join(folder(scan), scan)

    db.print(path)

    subj = db.patients(path)[0] 

    db.delete(subj + [DCE+'_Maps', 'S0_copy2'])

    # Copy within the database
    db.copy(subj + [DCE+'_Maps', 'S0'], subj + [DCE+'_Maps', 'S0_copy3'])
    # db.copy(subj + [DCE+'_Maps'], subj + [DCE+'_Maps_copy'])
    # db.copy(subj, [subj[0], subj[1] + '_Copy'])

    #subj1 = [os.path.join(folder(scan), 'NEW_DICOM'), subj[1]]
    #db.copy(subj + [DCE+'_Maps', ('S0',1)], subj1 + [DCE+'_Test', 'S02'])
    #db.copy(subj + [DCE+'_Maps'], subj1 + [DCE+'_Maps_Copy'])
    # db.copy(subj1 + [DCE+'_Maps_Copy'], subj + [DCE+'_Maps'])
    
    # series = subj + [DCE+'_Maps', 'S0']
    # vol = db.volume(series)
    # db.write_volume(vol, results + ['DummyS0_2'], ref=series)

    # series = subj + [SCANS, (DCE, 0)]
    # vol = db.volume(series, dims='TriggerTime')
    # db.write_volume(vol, results + ['DummyDCE'], ref=series)

    # dims = ('SliceLocation', 'InstanceNumber')
    # times = ((0x0009,0x10E9), 'TriggerTime') # GE specific
    # vals = db.unique(dims + times, series)

    db.print(path)



# Pipeline stages

def unzip_data(dataset):

    file = os.path.join(DATAPATH, dataset + '.zip')
    dicom_folder = os.path.join(folder(dataset), dataset)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(dicom_folder)
        print(f"✅ Extracted: {file} → {dicom_folder}")
    except zipfile.BadZipFile:
        print(f"❌ Error: {file} is not a valid ZIP file")
    except Exception as e:
        print(f"❌ Error extracting {file}: {e}")


def print_dicom_tree(scan):
    path = os.path.join(folder(scan), scan)
    db.print(path)

def merge_dynamics(scan, input=None, output=None):
    # Read time curves
    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    dyn = db.series(subj + [input], name=DCE)
    for series in tqdm(dyn, desc='Merging dynamics..'):
        db.copy(series, subj + [output, DCE])

def compute_maps(scan, input=None, output=None):

    # Load data
    path = os.path.join(folder(scan), scan)
    input = db.studies(path, contains=input)[0]
    output = input[:2] + [output]
    ref = input + [(DCE, 0)]

    # Compute maps
    series = db.series(input, name=DCE)
    for i, s in enumerate(series):
        vol = db.volume(s, dims='TriggerTime')
        arr = vol.values
        if i==0:
            S0 = np.mean(arr[...,:30], axis=-1)
            max_enh = np.amax(arr[...,:150], axis=-1) - S0
            avr = [np.mean(arr, axis=-1)]
            total = np.sum(arr, axis=-1)
            nt = arr.shape[-1]
        else:
            avr.append(np.mean(arr, axis=-1))
            total += np.sum(arr, axis=-1)
            nt += arr.shape[-1]
    auc = total - S0*nt

    # Save as dicom
    S0 = vreg.volume(S0, vol.affine)
    max_enh = vreg.volume(max_enh, vol.affine)
    auc = vreg.volume(auc, vol.affine)

    db.write_volume(S0, output + ['S0'], ref)
    db.write_volume(max_enh, output + ['MaxEnh'], ref)
    db.write_volume(auc, output + ['AUC'], ref)

    for i, avr_i in enumerate(avr):
        avr_i = vreg.volume(avr_i, vol.affine)
        db.write_volume(avr_i, output + [f'Mean{i+1}'], ref)


def segment_aorta(scan, input=None, output=None):

    maps = ['MaxEnh', 'S0']

    # Get data
    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    data = subj + [input]
    results = subj + [output]
    vols = [db.volume(data + [m]) for m in maps]

    # Perform segmentation
    mask = miblab.totseg(        
        vols, 
        cutoff=0,
        task="total_mr", 
        roi_subset=['aorta'], 
        quiet=True, 
        device='cpu',
    )

    # Save results to database
    for roi, vol in tqdm(mask.items(), desc='Writing to DICOM..'):
        db.write_volume(vol, results + [roi], ref=data + [maps[0]])


def segment_organs(scan, input=None, output=None):

    maps = ['MaxEnh', 'S0', 'Mean4']

    # Get data
    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    data = subj + [input]
    results = subj + [output]
    vols = [db.volume(data + [m]) for m in maps]

    # "thigh_shoulder_muscles_mr"
    #for task in ["total_mr", "vertebrae_mr", "body_mr", "liver_segments_mr"]:
    #for task in ["body_mr", "liver_segments_mr"]:
    for task in ["liver_segments_mr"]:

        # Perform segmentation
        mask = miblab.totseg(        
            vols, 
            cutoff=1e-3,
            task=task, 
            quiet=True, 
            device='cpu',
        )
        # Save results to database
        for roi, vol in tqdm(mask.items(), desc='Writing to DICOM..'):
            db.write_volume(vol, results+[roi], ref=data+[maps[0]])


def export_time_curves(scan, input=None):
                       
    dynamics, rois = input
    resultspath = os.path.join(folder(scan), 'dmr')
    path = os.path.join(folder(scan), scan)

    dims = ('SliceLocation', 'InstanceNumber')
    tstart, tacq = (0x0009,0x10E9), 'TriggerTime' # GE specific
    subj = db.patients(path)[0]
    
    # Read masks
    mask = {}
    data = {
        'tacq': ['Acquisition time', 'min', 'float'],
    }
    sig = {
        'tacq': [],
    }
    roi_subset = db.series(subj + [rois])
    for roi in tqdm(roi_subset, desc='Reading masks..'):
        img, _ = db.pixel_data(roi, dims='SliceLocation')
        mask[roi[-1]] = img >= 0.5
        data[roi[-1]] = [f'Signal in {roi[-1]}', 'a.u.', 'float']
        sig[roi[-1]] = []
    
    # Read time curves
    dyn = db.series(subj + [dynamics], name=DCE)
    for series in tqdm(dyn, desc='Reading time curves..'):
        s, _, t = db.pixel_data(series, dims, include=(tstart, tacq)) 
        sig['tacq'] += list(t[tstart][0,0] + t[tacq][0,:] / 1000)
        for roi in tqdm(mask, desc='Extracting signal averages..'):
            sig[roi] += [
                np.mean(s[...,k][mask[roi]]) 
                for k in range(t[tstart].shape[1])
            ]

    # Convert time to minutes
    sig['tacq'] = [t/60. for t in sig['tacq']]

    # Save as dmr
    subj = scan[:9]
    study = scan[-11:]
    data = {
        'data': data, 
        'rois': {
            subj: {
                study: sig,
            },
        },
    }
    file = os.path.join(resultspath, subj + '_' + study)
    pydmr.write(file, data, 'nest')
    


def create_dce_zarr(scan, input=None):

    store = os.path.join(folder(scan), 'disco.zarr')
    if os.path.exists(store):
        shutil.rmtree(store)

    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    series = db.series(subj + [input], name=DCE)
    for i, s in tqdm(enumerate(series), desc='Series to zarr..'):
        vol = db.volume(s, dims='TriggerTime')
        if i==0:
            zarray = vreg.create_zarr(
                vol, 
                shape=vol.shape, 
                dtype='float32', 
                chunks=vol.shape[:2] + (1,1),
                store=store,
            )
        else:
            zarray.append(vol.values, axis=3)


def zarr_moco(scan):

    subj = scan[:9]
    study = scan[-11:]

    roidata = os.path.join(folder(scan), 'dmr', subj + '_' + study)
    rois = pydmr.read(roidata)['rois']

    discodata = os.path.join(folder(scan), 'disco.zarr')
    array = zarr.open(discodata, mode='r')

    mdreg.fit(
        array,
        fit_image={
            'func': mdreg.fit_2cm_lin,
            'time': rois[subj, study, 'tacq'][:array.shape[-1]],
            'aif': rois[subj, study, 'aorta'][:array.shape[-1]],
            'baseline': 20,
            'parallel': False,
            'progress_bar': True,  
        },
        fit_coreg={
            'package': 'ants',
            'type_of_transform': 'SyNOnly',
            'parallel': False,
            'progress_bar': True,
        },
        maxit=3,
        path=os.path.join(folder(scan), 'disco_moco'),
        verbose=2,
    )


def zarr_to_dicom(scan, output=None):

    zarrays = os.path.join(folder(scan), 'disco_moco')
    coreg = zarr.open(os.path.join(zarrays, 'coreg.zarr'), mode='r')
    fit = zarr.open(os.path.join(zarrays, 'fit.zarr'), mode='r')

    dicom = os.path.join(folder(scan), scan)
    subj = db.patients(dicom)[0]
    results = subj + [output]   
    
    # Save dynamics in original 4 series format
    i0=0
    refs = db.series(subj + [SCANS], name=DCE)
    for i, ref in tqdm(enumerate(refs), desc='zarr to DICOM..'):
        vol = db.volume(ref, dims='TriggerTime') 

        i1 = i0 + vol.shape[-1]
        
        vol.set_values(coreg[:, :, :, i0: i1])
        db.write_volume(vol, results + ['DISCO_moco'], ref)
        vol.set_values(fit[:, :, :, i0: i1])
        db.write_volume(vol, results + ['DISCO_fit'], ref)

        i0 = i1

    # Save parameters to dicom
    pars = zarr.open(os.path.join(zarrays, 'pars.zarr'), mode='r')

    Fp = vreg.volume(pars[:,:,:,0], vol.affine)
    Tp = vreg.volume(pars[:,:,:,1], vol.affine)
    PS = vreg.volume(pars[:,:,:,2], vol.affine)
    Te = vreg.volume(pars[:,:,:,3], vol.affine)

    db.write_volume(Fp, results + ['DISCO_Fp'], ref)
    db.write_volume(Tp, results + ['DISCO_Tp'], ref)
    db.write_volume(PS, results + ['DISCO_PS'], ref)
    db.write_volume(Te, results + ['DISCO_Te'], ref)



def map_molli(scan, input=None, output=None):

    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    results = subj + [output]

    mollis = db.series(subj + [input], contains=MOLLI)
    for i, molli in tqdm(enumerate(mollis), desc=f'Fitting MOLLI'):

        vol = db.volume(molli, dims='InversionTime')

        coreg, fit, transfo, pars = mdreg.fit(
            vol.values[:,:,0,:], 
            fit_image = {
                'func': mdreg.fit_abs_exp_recovery_2p,
                'TI': vol.coords[0,:]/1000,
                'bounds':([0,0], [np.inf,3]),
                'parallel': False,
                'progress_bar': True,
            }, 
            fit_coreg = {
                'package': 'skimage',
                'attachment': 30,
                'parallel': False,
                'progress_bar': True,
            },
            maxit=3, 
            verbose=2,
        )

        # Create output volumes
        S0 = vreg.volume(np.expand_dims(pars[:,:,0], 2), vol.affine)
        T1 = vreg.volume(np.expand_dims(pars[:,:,1], 2), vol.affine)
        coreg = vreg.volume(np.expand_dims(coreg, 2), vol.affine, vol.coords, vol.dims)
        fit = vreg.volume(np.expand_dims(fit, 2), vol.affine, vol.coords, vol.dims)

        # Save results as DICOM
        db.write_volume(S0, results + [f'MOLLI_S0_{i+1}'], ref=molli)
        db.write_volume(T1, results + [f'MOLLI_T1_{i+1}'], ref=molli)
        db.write_volume(coreg, results + [f'MOLLI_coreg_{i+1}'], ref=molli)
        db.write_volume(fit, results + [f'MOLLI_fit_{i+1}'], ref=molli)


def align_molli_moco(scan, input=None, output=None):

    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    data = subj + [input]
    results = subj + [output]

    # Optimizer settings
    opt = {
        'method': 'brute',
        'grid': (
            [-20, 20, 10],
            [-20, 20, 10],
            [-20, 20, 10],
        ),
        'progress': True,
    }
    disco = db.volume(subj + [DCE + '_maps_moco', 'Mean1'])

    for i in [0,1]:

        S0_name = f'MOLLI_S0_{i+1}'
        T1_name = f'MOLLI_T1_{i+1}'
        coreg_name = f'MOLLI_coreg_{i+1}'

        # Get the volumes
        S0 = db.volume(data + [S0_name])
        T1 = db.volume(data + [T1_name])
        coreg = db.volume(data + [coreg_name], dims='InversionTime')

        # Perform the coregistration
        shift = S0.find_translate_to(disco, optimizer=opt, coords='volume')
        
        # Write results to dicom
        S0 = coreg.translate(S0, coords='volume')
        T1 = coreg.translate(T1, coords='volume')
        coreg = coreg.translate(shift, coords='volume')

        db.write_volume(S0, results+[S0_name], ref=data+[S0_name])
        db.write_volume(T1, results+[T1_name], ref=data+[T1_name])
        db.write_volume(coreg, results+[coreg_name], ref=data+[coreg_name])


def export_molli_moco_data(scan, input=None):

    maps, rois = input
    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]

    # Define parameters
    pars = {}
    data = {}
    roi_subset = db.series(subj + [rois], isin=ROI_MOLLI)
    for roi in tqdm(roi_subset, desc='Reading masks..'):
        mask = db.volume(roi)
        for i in [0,1]:
            T1 = db.volume(maps + [f'MOLLI_T1_{i+1}'])
            locs = mask.slice_like(T1).values > 0.5
            pars[f'T1_{roi[-1]}_{i+1}'] = np.mean(T1.values[locs])
            data[f'T1_{roi[-1]}_{i+1}'] = [f'T1 of scan {i+1} in {roi[-1]}', 'sec', 'float']

    # Save as dmr
    subj = scan[:9]
    study = scan[-11:]
    data = {
        'data': data, 
        'pars': {
            subj: {
                study: pars,
            },
        },
    }
    resultspath = os.path.join(folder(scan), 'dmr')
    file = os.path.join(resultspath, subj + '_' + study + '_molli')
    pydmr.write(file, data, 'nest')


def export_volumes(scan, input=None):

    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]

    # Define parameters
    pars = {}
    data = {}
    roi_subset = db.series(subj + [input], isin=ROI_SUBSET)
    for roi in tqdm(roi_subset, desc='Reading masks..'):
        mask = db.volume(roi)
        nvox = np.count_nonzero(mask.values > 0.5)
        vol = f'volume_{roi}'
        pars[vol] = nvox * np.prod(mask.spacing) / 1000
        data[vol] = [f'Volume of {roi}', 'mL', 'float']

    # Save as dmr
    subj = scan[:9]
    data = {
        'data': data, 
        'pars': {
            subj: {
                '1scan': pars,
            },
        },
    }
    resultspath = os.path.join(folder(scan), 'dmr')
    file = os.path.join(resultspath, subj + '_volumes')
    pydmr.write(file, data, 'nest')


def export_params(scan, input=None):

    pars = ['RepetitionTime', 'EchoTime', 'FlipAngle', 'PatientWeight', 'PatientSize']
    path = os.path.join(folder(scan), scan)
    subj = db.patients(path)[0]
    vals = db.unique(pars, subj + [input, (DCE, 0)])

    # Define parameters
    data = {
        'TR': ['Repetition time', 'sec', 'float'], 
        'FA_1': ['Flip angle - first scan', 'deg', 'float'], 
        'TE': ['Echo time', 'sec', 'float'], 
        'weight': ['Subject body weight', 'kg',	'float'],
        'size': ['Subject length', 'm',	'float'], 
        'dose_1': ['Contrast agent dose - first scan', 'mL/kg', 'float'], 
        't0': ['Baseline duration', 'sec', 'float'],
    }
    pars = {
        'TR': vals['RepetitionTime']/1000,
        'TE': vals['EchoTime']/1000,
        'FA_1': vals['FlipAngle'], 
        'weight': vals['PatientWeight'],
        'size': vals['PatientSize'],
        'dose_1': 0.025, # with some exceptions
        't0': 60,
    }

    # Save as dmr
    subj = scan[:9]
    data = {
        'data': data, 
        'pars': {
            subj: {
                '1scan': pars,
            },
        },
    }
    resultspath = os.path.join(folder(scan), 'dmr')
    file = os.path.join(resultspath, subj + '_params')
    pydmr.write(file, data, 'nest')


def build_export(scan):

    resultspath = os.path.join(folder(scan), 'dmr')
    subj = scan[:9]
    sourcefiles = [
        os.path.join(resultspath, subj + '_moco'),
        os.path.join(resultspath, subj + '_molli'),
        os.path.join(resultspath, subj + '_params'),
        os.path.join(resultspath, subj + '_volumes'),
    ]
    exportfile = os.path.join(resultspath, subj + '_export')
    pydmr.concat(sourcefiles, exportfile)






def onescan(dataset):

    start_time = time.time()

    # # Prepare data
    # # ------------
    # unzip_data(dataset)

    # # Get AIF for mdreg
    # # -----------------
    
    compute_maps(dataset, input=SCANS, output=DCE+'_Maps')
    # segment_aorta(dataset)
    # export_time_curves(dataset)

    # # Perform motion correction
    # # -------------------------
    # create_dce_zarr(dataset)
    # zarr_moco(dataset)
    # zarr_to_dicom(dataset)

    # # Segment and get time curves
    # # ---------------------------
    # compute_mean(dataset, moco=True)
    # segment_mean(dataset, moco=True)
    # export_time_curves(dataset, moco=True)

    # # Get T1 values
    # # -------------
    # map_molli(dataset)
    # align_molli_moco(dataset)
    # export_molli_moco_data(dataset)

    # # Build final export
    # # ------------------
    # export_volumes(database, resultspath)
    # export_params(database, resultspath)
    # build_export(database, resultspath)
    
    # # Display results
    # # ---------------
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    #dicom_folder = os.path.join(folder(dataset), dataset)
    #display(dicom_folder)


def twoscan(dataset):

    start_time = time.time()

    dummy(dataset)

    # # Prepare data
    # # ------------
    # unzip_data(dataset)
    # print_dicom_tree(dataset)
    # merge_dynamics(dataset, input=SCANS, output=DCE)

    # # Get AIF for mdreg
    # # -----------------
    # compute_maps(dataset, input=SCANS, output=DCE+'_Maps')
    # segment_aorta(dataset, input=DCE + '_Maps', output='Aorta ROIs')

    # TODO: Add interactive checkpoint

    # export_time_curves(dataset, data=(SCANS, 'Aorta ROIs'))

    # # Perform motion correction
    # # -------------------------
    # create_dce_zarr(dataset, input=SCANS)
    # zarr_moco(dataset)
    # zarr_to_dicom(dataset, output='Moco')

    # Segment and get time curves
    # ---------------------------
    # compute_maps(dataset, input='Moco', output=DCE + '_moco_maps')
    # segment_organs(dataset, input='DescriptiveMapsMoco', output='MocoSegmentations')
    
    # export_time_curves(dataset, input=('Moco', 'Aorta ROIs'))

    # # Get T1 values
    # # -------------
    # map_molli(dataset, input=SCANS, output='MOLLI maps')
    # align_molli_moco(dataset, input='MOLLI maps', output='MOLLI maps aligned')
    # export_molli_moco_data(dataset, input=('MOLLI maps aligned', 'MocoSegmentations'))

    # # Build final export
    # # ------------------
    # export_volumes(database, input='MocoSegmentations')
    # export_params(database, input=SCANS)
    # build_export(database, resultspath)
    
    # # Display results
    # # ---------------
    print("--- %s minutes ---" % ((time.time() - start_time)/60))


def visit(dataset):

    start_time = time.time()
    
    build_visit_dicom(dataset)

    # # Display results
    # # ---------------
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    #dicom_folder = os.path.join(visit_results(dataset), 'DICOM')
    #display(dicom_folder)



if __name__ == '__main__':

    # onescan('MEDCIC_02_Visit1Scan1')
    twoscan('MEDCIC_02_Visit1Scan2')
    # visit('MEDCIC_02_Visit1')



