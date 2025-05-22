import os, pathlib, multiprocessing as mp


class Config:
    # --------- paths ----------
    input_dir = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes/20231117161658")
    output_dir = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes_masked/20231117161658")
    model_dir = "./models/nnUNetTrainerV2__nnUNetPlans__2d"
    temp_dir = "/dev/shm/nnunet_tmp"  # fast RAM disk (change or remove)

    # --------- pipeline knobs ----------
    downscale_factor = 4
    threshold = 0.5
    n_preproc = 24  # CPU workers that read & down-scale
    n_postproc = 12  # CPU workers that up-scale & save
    queue_size = 128  # how many slices may wait in RAM

    # --------- nnUNet tuning ----------
    tile_step_size = 0.75  # larger ⇒ fewer tiles ⇒ faster
    use_mirroring = False  # test-time aug; True = slower but tiny accuracy gain
