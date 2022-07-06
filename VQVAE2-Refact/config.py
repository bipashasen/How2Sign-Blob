DATASETS_MAP = {
    'hand2gesture': 1,
    'blob2full': 2,
    'hand2gesture4pixelsnail': 3,
    'facetranslationmultipleframes': 4, 
    'facetranslation': 5
}

DATASET = 15
LATENT_LOSS_WEIGHT = 1 # latent loss has been kept at 1
PERCEPTUAL_LOSS_WEIGHT = 0.25 # modified for the experiment training from scratch

# weights corresponding to mocoganhd discriminator 
G_LOSS_2D_WEIGHT = 0.25
G_LOSS_3D_WEIGHT = 0.25

image_disc_weight = 0.5
video_disc_weight = 0.5

D_LOSS_WEIGHT = 0.1

SAMPLE_SIZE_FOR_VISUALIZATION = 8
DISC_LOSS_WEIGHT = 0.25 # TODO - modify

# ---- config for vlog_all5losses -- video-predisc ---- #
# adversarial_disc_start = 15000
# recon_weight = 1.0

# # max(min(source_reconstruction_weight_after_disc_start/global_step, 1.0), 0.02)
# adaptive_source_rec_weight_start = 1000
# # source_reconstruction_weight_after_disc_start = 1000

# target_by_source_mask=False
# loss_with_no_enlargement=True

# save_folder = 'vlog_all5losses_predisc_ptrt'
# checkpoint_suffix = save_folder

# perceptual_loss = True

# ---- config for vlog_all5losses -- 1frame ---- #
# adversarial_disc_start = 200000
# recon_weight = 1.0

# # max(min(source_reconstruction_weight_after_disc_start/global_step, 1.0), 0.02)
# adaptive_source_rec_weight_start = 1000*5
# # source_reconstruction_weight_after_disc_start = 1000

# target_by_source_mask=False
# loss_with_no_enlargement=True

# save_folder = 'vlog_all5losses_1frame_nback/validation'
# checkpoint_suffix = save_folder

# perceptual_loss = True

# ---- config for vlog_all5losses -- video ---- #
# adversarial_disc_start = 59393
# recon_weight = 1.0

# # max(min(source_reconstruction_weight_after_disc_start/global_step, 1.0), 0.02)
# adaptive_source_rec_weight_start = 1000
# # source_reconstruction_weight_after_disc_start = 1000

# target_by_source_mask=False
# loss_with_no_enlargement=True

# save_folder = 'vlog_all5losses'
# checkpoint_suffix = save_folder

# perceptual_loss = True