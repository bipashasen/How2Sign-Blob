# sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/VQVAE2-FaceMultiFrames-Refact'
sample_folder = '/ssd_scratch/cvit/aditya1/vqvae2_perturbed_singleframe'
BASE_PATH = '/ssd_scratch/cvit/aditya1'

datasets = {
    'hand2gesture': 1,
    'blob2full': 2,
    'hand2gesture4pixelsnail': 3,
    'facetranslationmultipleframes': 4, 
    'facetranslationsingleframe': 5,
}

dataset = 5
# for loss calculation
latent_loss_weight = 0.25
# for visualization
sample_size = 8