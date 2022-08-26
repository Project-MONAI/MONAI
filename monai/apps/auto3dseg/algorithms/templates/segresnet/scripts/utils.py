
def roi_ensure_divisible(roi_size, levels):
    '''
    Calculate the cropping size (roi_size) that is evenly divisible by 2 for the given number of hierarchical levels
    e.g. for for the network of 5 levels (4 downsamplings), roi should be divisible by 2^(5-1)=16 
    '''

    multiplier = pow(2, levels-1)
    roi_size2=[]
    for r in roi_size:
        if r % multiplier !=0:
            p = multiplier * max(2, int(r/ float(multiplier))) #divisible by levels, but not smaller then 2 at final level
            roi_size2.append(p)
        else:
            roi_size2.append(r)
    
    return roi_size2

def roi_ensure_levels(levels, roi_size, image_size):
    '''
    In case the image (at least one axis) is smaller then roi, reduce the roi and number of levels 
    '''

    while all([r > 1.5*i for r,i in zip(roi_size, image_size)]) and levels>1:
        levels = levels-1
        roi_size = [r//2 for r in roi_size]
    return levels, roi_size