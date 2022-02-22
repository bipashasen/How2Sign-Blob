sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/VQVAE2-FaceMultiFrames-Refact'

datasets = {
    'hand2gesture': 1,
    'blob2full': 2,
    'hand2gesture4pixelsnail': 3,
    'facetranslationmultipleframes': 4, 
}

dataset = 4
# for loss calculation
latent_loss_weight = 0.25
# for visualization
sample_size = 8