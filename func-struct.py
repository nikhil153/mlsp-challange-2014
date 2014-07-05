import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# import data
AAL = nib.load('aal_labels.nii')
FNC = nib.load('rs_fMRI_ica_maps.nii').get_data()
SBM = nib.load('gm_sMRI_ica_maps.nii').get_data()

dims = AAL.shape
AAL = AAL.get_data()

AAL = AAL.reshape(dims[0]*dims[1]*dims[2], )
FNC = FNC.reshape(dims[0]*dims[1]*dims[2], 28)
SBM = SBM.reshape(dims[0]*dims[1]*dims[2], 32)

# scale data to [-1, 1]
FNC = FNC / np.max((np.abs(np.min(FNC)), np.abs(np.max(FNC)))) 
SBM = SBM / np.max((np.abs(np.min(SBM)), np.abs(np.max(SBM)))) 

# init output array
OUT = np.ones((28, 32, 116))

# loop through atlas regions
for R in [ROI for ROI in np.unique(AAL) if ROI > 0]:
    
    # record ROI indicies
    idx = np.where(AAL == R)[0]

    # loop through functional data
    for F in np.arange(FNC.shape[1]):
        
        # grab the loadings from component F
        F_data = FNC[idx, F]

        # loop through structural data
        for S in np.arange(SBM.shape[1]):

            # grab the loadings from component S
            S_data = SBM[idx, S]

            # subtract sum F from sum S, store.
            OUT[F, S, R-1] = np.sum(S_data) - np.sum(F_data) 

# plot jam
for x in np.arange(116):
    plt.subplot(9, 13, x+1)
    plt.imshow(OUT[:, :, x], cmap=plt.cm.RdBu_r, 
                             interpolation='nearest',
                             vmin=-1,
                             vmax=1)
    plt.axis('off')
plt.tight_layout()
plt.show()

