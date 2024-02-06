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






