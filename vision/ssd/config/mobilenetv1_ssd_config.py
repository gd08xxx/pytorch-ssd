import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 2048
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(128, 16, SSDBoxSizes(25, 50), [2, 3]),
    SSDSpec(64, 32, SSDBoxSizes(80, 140), [2, 3]),
    SSDSpec(32, 64, SSDBoxSizes(200, 500), [2, 3]),
    SSDSpec(16, 128, SSDBoxSizes(1105.92, 1454.08), [2, 3]),
    SSDSpec(8, 256, SSDBoxSizes(1454.08, 1802.24), [2, 3]),
    SSDSpec(4, 512, SSDBoxSizes(1802.24, 2150.4), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)


def set_image_size(size=2048, min_ratio=20, max_ratio=90):
    global image_size
    global specs
    global priors
    
    from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
    
    import torch
    import math
    import logging
        
    image_size = size
    ssd = create_mobilenetv1_ssd(num_classes=3) # TODO does num_classes matter here?
    x = torch.randn(1, 3, image_size, image_size)
    
    feature_maps = ssd(x, get_feature_map_size=True)
    
    steps = [
        math.ceil(image_size * 1.0 / feature_map) for feature_map in feature_maps
    ]
    step = int(math.floor((max_ratio - min_ratio) / (len(feature_maps) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(image_size * ratio / 100.0)
        max_sizes.append(image_size * (ratio + step) / 100.0)
    min_sizes = [image_size * (min_ratio / 2) / 100.0] + min_sizes
    max_sizes = [image_size * min_ratio / 100.0] + max_sizes
    
    # this update logic makes different boxes than the original for 300x300 (but better for power-of-two)
    # for backwards-compatibility, keep the default 300x300 config if that's what's being called for
    if image_size != 2048:
        specs = []
        
        for i in range(len(feature_maps)):
            specs.append( SSDSpec(feature_maps[i], steps[i], SSDBoxSizes(min_sizes[i], max_sizes[i]), [2, 3]) )   # ssd-mobilenet-* aspect ratio is [2,3]

    logging.info(f'model resolution {image_size}x{image_size}')
    for spec in specs:
        logging.info(str(spec))
    
    priors = generate_ssd_priors(specs, image_size)
    
#print(' ')
#print('SSD-Mobilenet-v1 priors:')
#print(priors.shape)
#print(priors)
#print(' ')

#import torch
#torch.save(priors, 'mb1-ssd-priors.pt')

#np.savetxt('mb1-ssd-priors.txt', priors.numpy())
