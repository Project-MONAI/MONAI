# modified from sources: 
# - Original implementation from Macenko paper in Matlab: https://github.com/mitkovetta/staining-normalization
# - Implementation in Python: https://github.com/schaugf/HEnorm_python
import openslide
import cupy as cp
from PIL import Image
from typing import Tuple
from monai.transforms.transform import Transform


class StainNormalizer(Transform):
    """
    Stain Normalize patches of a digital pathology image. Performs Stain Deconvolution using the Macenko method.
    A source patch can be normalized using a reference stain matrix, or using a target image from
    which a target stain is extracted. For using the reference stain, run only the normalize_patch function.
    To use the stain from a target image, run extract_stain first to modify the target stain matrix used, then
    run normalize_patch on each patch to be stain normalized.
    
    Args:
        Io: (optional) transmitted light intensity
        alpha: (optional) tolerance for the pseudo-min and pseudo-max
        beta: (optional) OD threshold for transparent pixels
        target_image: (optional) OpenSlide image to perform stain deconvolution of,
            to obtain target stain matrix
        
    """
    def __init__(self, Io: float=240, alpha: float=1, beta: float=0.15, target_image: openslide.OpenSlide=None) -> None:
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        
        # reference maximum stain concentrations for H&E
        self.maxCRef = cp.array([1.9705, 1.0308])
        
        # target H&E stain is set to reference H&E OD matrix 
        self.target_HE = cp.array([[0.5626, 0.2159],
                    [0.7201, 0.8012],
                    [0.4062, 0.5581]])
        if target_image!=None:
            self._extract_stain(target_image)
    
    def _stain_deconvolution(self, img: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Perform Stain Deconvolution using the Macenko Method.
        
        Args:
            img: image to perform stain deconvolution of
            
        Return:
            HE: H&E OD matrix for the image (first column is H, second column is E)
            C2: stain concentration matrix for the input image 
        """
        # define height and width of image
        h, w, c = img.shape
        
        # RGBA to RGB
        img = img[:, :, :-1]
        
        # reshape image
        img = img.reshape((-1,3))
        
        # calculate optical density
        OD = -cp.log((img.astype(cp.float)+1)/self.Io)
        
        # remove transparent pixels
        ODhat = OD[~cp.any(OD<self.beta, axis=1)]
        
        # compute eigenvectors
        eigvals, eigvecs = cp.linalg.eigh(cp.cov(ODhat.T))
        
        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues    
        That = ODhat.dot(eigvecs[:,1:3])
        
        # find the min and max vectors and project back to OD space
        phi = cp.arctan2(That[:,1],That[:,0])
        minPhi = cp.percentile(phi, self.alpha)
        maxPhi = cp.percentile(phi, 100-self.alpha)
        vMin = eigvecs[:,1:3].dot(cp.array([(cp.cos(minPhi), cp.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(cp.array([(cp.cos(maxPhi), cp.sin(maxPhi))]).T)
        
        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = cp.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = cp.array((vMax[:,0], vMin[:,0])).T
        
        # rows correspond to channels (RGB), columns to OD values
        Y = cp.reshape(OD, (-1, 3)).T
        
        # determine concentrations of the individual stains
        C = cp.linalg.lstsq(HE,Y, rcond=None)[0]
        
        # normalize stain concentrations
        maxC = cp.array([cp.percentile(C[0,:], 99), cp.percentile(C[1,:],99)])
        tmp = cp.divide(maxC,self.maxCRef)
        C2 = cp.divide(C,tmp[:, cp.newaxis])
        return HE, C2
    
    def _extract_stain(self, target_image: openslide.OpenSlide) -> None:
        """Extract a reference stain from a target image.
        
        To extract reference stain, the image at the highest level (the level with
        lowest resolution) is used. Then, stain deconvolution provides the stain matrix.
        
        Args:
            target_image: (optional) OpenSlide image to perform stain deconvolution of,
                to obtain target stain matrix
        """
        highest_level = target_image.level_count - 1
        dims = target_image.level_dimensions[highest_level]
        target_image_at_level = target_image.read_region((0,0), highest_level, dims)
        target_img = cp.array(target_image_at_level)
        self.target_HE, _ = self._stain_deconvolution(target_img)

    def __call__(self, data: cp.ndarray) -> cp.ndarray:
        """Normalize a patch to a reference / target image stain.
        
        Performs stain deconvolution of the patch to obtain the stain concentration matrix
        for the patch. Then, performs the inverse Beer-Lambert transform to recreate the 
        patch using the target H&E stain. 
        
        Args:
            patch: image patch to stain normalize
            
        Return:
            patch_norm: normalized patch
        """
        h, w, _ = data.shape
        _, patch_C = self._stain_deconvolution(data)
        
        patch_norm = cp.multiply(self.Io, cp.exp(-self.target_HE.dot(patch_C)))
        patch_norm[patch_norm>255] = 254
        patch_norm = cp.reshape(patch_norm.T, (h, w, 3)).astype(cp.uint8)
        return patch_norm