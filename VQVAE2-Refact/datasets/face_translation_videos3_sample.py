def generate_sample():
    source_frames_sampled = ['0uNEEfQKfwk/00000/00418.jpg']
    target_frames_sampled = ['0uNEEfQKfwk/00194/00046.jpg']
    
    source_image_path = source_frames_sampled[0]
    target_image_path = target_frames_sampled[0]
    SOURCE_BASE_PATH = osp.join(source_image_path.rsplit('.', 1)[0])
    TARGET_BASE_PATH = osp.join(target_image_path.rsplit('.', 1)[0])

    source_landmark_npz = SOURCE_BASE_PATH + '_landmarks_compressed.npz'
    target_landmark_npz = TARGET_BASE_PATH + '_landmarks_compressed.npz'

    # read the data and apply framewise transformation
    source_face_transformed, source_background_mask_transformed, \
    target_image, target_face_mask, perturbed_image = \
            generate_aligned_image(source_landmark_npz, target_landmark_npz, 
            source_image_path, target_image_path, poisson_blend_required = False)

    plt.figure()
    plt.title('Combination using alignment')
    plt.imshow(perturbed_image)

    
    source_face_transformed, source_background_mask_transformed, \
    target_image, target_face_mask, perturbed_image = \
            generate_warped_image(source_landmark_npz, target_landmark_npz, 
            source_image_path, target_image_path, poisson_blend_required = False,
                                  require_full_mask = False)
    
    
    plt.figure()
    plt.title('Combination using warping')
    plt.imshow(perturbed_image)
    
    target_background = apply_mask(np.invert(source_background_mask_transformed), target_image)
    
    plt.figure()
    plt.title('Target background using source mask')
    plt.imshow(target_background)
    
    target_background = apply_mask(target_face_mask, target_image)
    
    plt.figure()
    plt.title('Target background using target mask')
    plt.imshow(target_background)
    
generate_sample()