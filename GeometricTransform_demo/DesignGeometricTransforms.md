## Design for Geometric Transforms
### Introduction
In designing and extending the MONAI API for Geometric transforms, we need to be careful about 2 things
That the external API user experience remains unchanged. I believe that many people will not use Geometric Transforms right away, so of course we should be mindful to continue maintaining the  current user-facing API.

I believe that we should write separate transforms for Geometric objects. To begin with, these are the following transforms, we should be working with
- `Rotate`
- `Flip`
- `Resample`
- `Zoom`
- 'Identity` transforms (which do nothing)

I think these 4 will cover 80 percent of the cases.

I will focus more on transparency and readability than a very efficient and tightly integrated code.

We should get user feedback by introducing changes slowly and see what kind of feedback we get before making a lot of tightly integrated changes.

### Geometry Preprocessing design
I do not think that the Geometry objects should be given First Class treatments. The reasons are as follows:
- The only reason Geometric Points exist in MONAI is because of annotation on images and its relationship with the image. They do not exist independently.
- If there are standalone Geometric objects like meshes, there are libraries like pytorch_geometric which can handle them more efficiently and use Graph Neural Networks to train.

The main purpose of supporting Geometric objects in MONAI is to able to handle annotation. I also feel that the major types of annotations that we need to handle are as follows:
- Polygons on 2D images
- Bounding box on 2D and 3D images.
This should cover most of the requirements. Anything else is an overkill at the moment. If we can handle point transformations, that should suffice as for polygons and even meshes it is just a matter of maintaining the list of point indices.

### Ideal user-facing API
#### Loading Data
For loading data. So far we were only concerned with images and image segmentation so we used the function `LoadImaged(keys=[‘image’, ‘label’])`. This was fine until now as both the image and label were ultimately images. Instead of `LoadImaged` I suggest using the following function
```
LoadDatad(keys=[“image”, “label”, “geom”])
```
`LoadDatad` is a wrapper function that directs the loading of `image` and `labels` to the existing functions and directs the `geom` loading to a `LoadPointsd` function.

**Pros**: We can maintain the underlying functionality with the transform chain for images but add in new transform function for points. I am saying LoadDatad to be a more generic functional wrapper that can load images, segmentation as well as geometric points, and maybe more non-pixel data in the future (like text or EKG data).

In addition, we need to have a separate dictionary for the labels inside geometric labels for example different groups of points referring to different types of annotation. We will need a mechanism to differentiate and identify point annotations for two different entities like a bounding box for a lung nodule, vs a polygon for a scar tissue.

#### Compose
A proposal for `Compose` may look like
```
Compose[LoadDatad(keys=[‘image’,‘geom’]),
        Flipd(keys=[‘image’,‘geom’]),
		    ScaleIntensityRanged(keys=[’image’,‘geom’]),
		    ResampleImaged(keys=[‘image’, ‘geom’])
```
In this case, we won't need to change anything. Whenever there is a geom key in the list of keys we can direct our point transforms to another function like `FlipPoints`, `ResamplePoints` etc.

The reason for doing so is that we do not unintentionally break the current workflow and in the case while doing a sanity check we come across some weird point transformation that is not giving the expected result, the user will be able to write their own transforms which can be overloaded or called in place of the existing ones.

I forsee this happening, as image-annotation alignment has a lot of moving parts that cannot be often predicted. For example,
- Geometry and coordinate system interpretation in the annotation tool might be very different as to what we are used to viewing in ITKSnap or similar tools.
- Infact, there might be cases like in DicomRT where there might be more discrepancies between annotations coming from different viewers like OHIF, Siemens and others.
- Some people might be labeling jpeg images using a plethora of existing jpeg annotation software and they all might be interpreting the coordinate system a bit differently.

I saw that there was a suggestion for `TransformToImage` to assign a coordinate system to a set of points to address the above-mentioned problems but it may not be sufficient to alleviate all the changes.

The main thing I am shooting for here is that we should have separate array-based implementations for points which is clear and easy for a new user to edit and understand, at least initially.

As we roll this out, we can slowly start tightening the integration and chaining transformation matrices.
![Untitled presentation](https://github.com/vikashg/MONAI/assets/3863212/b6b7bb5a-f317-46e6-a7de-9b4fa342f6a8)

Similar to the above dataflow graph for loading images and points, we can have a data flow graph for transformations for points and images, where we implement point transformations.

## A Middle ground
I think we should have a mechanism for users to bring in their point transforms **cleanly** and they should mesh well with the existing workflow and or be able to overload the existing point transform we should be good.
Of course, in practice, the users can go inside the code and change things if it does not suit their requirements, but that can be very tedious considering how tightly transforms are being integrated. I would love to have a `contrib` folder as was the case for TensorFlow in its early days, where users were writing their own data loaders and functions. 
### A sample use case
Please have a look at my `RotateImageandAnnotations` here
```
class RotateImageAndAnnotations:
    def __init__(self, angle: float = 10.0):
        self.angle = angle

    def __call__(self, sample, *args, **kwargs):
        self.image = sample['images']
        self.label = sample['labels']
        rotated_image, rotated_points = self._rotate_image_points()
        return {'images': rotated_image, 'labels': rotated_points}

    def _rotate_image_points(self):
        h, w = self.image.shape[0], self.image.shape[1]
        # print(h,w)
        cX, cY = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        calculated_point_list = []

        for _points in self.label:
            corners = _points.reshape(-1, 2)
            corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
            new_points = np.dot(M, corners.T).T
            calculated_point_list.append(new_points)

        _image = ToNumpy()(self.image) # Ask about this added this to use warp affine
        rotated_image = cv2.warpAffine(_image, M, (nW, nH), flags=cv2.INTER_LINEAR)
        _rotated_image = ToTensor()(rotated_image)
        return _rotated_image, calculated_point_list
```
This is a special use case for rotation. As we all know rotating an image changes its field of view and it cuts off portions of the images around the corners if we are not careful. One way to avoid that is first padding the image, rotating and then resampling/cropping the image. Most likely the correct way to implement the same is rotating the image, use some trigonometry to find out the new size of the image and then resample the image and points based on this new image size. 

The point I am trying to make is that this is a specific use case and maybe applicable for pathology slides (where the whole image is important). In most general implementations, we do not need to do such thing. So, if there is a **clean** mechanism to build and integrate your own transforms as per your specific use case, that will be immensly helpful. We can have proper documentation for I/O for user-defined transforms and overloading processes. 
So far, we were able to get away with this because of the following reason
- We were only using Nifti and dicom for images which have pretty standard reading and writing mechanism.
- Generally, we all agreed on image transforms (more or less).
- Even if there were weird dicom images, the general (unwritten) practice I followed was to convert dicoms to Nifti or numpy before feeding them for training.
But going forward we should allow for user-defined transforms that can overload the existing transforms if needed **cleanly**. 

