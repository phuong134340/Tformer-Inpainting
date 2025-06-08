import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
from util.visualizer import Visualizer
import torch
from PIL import Image
import numpy as np


from comet_ml import Experiment  # Thêm import comet

if __name__ == '__main__':
    # Khởi tạo Comet experiment
    experiment = Experiment(
        api_key="bKMwHTWGZ1fqpCdlAWHrPrP1M",  # Thay bằng API key của bạn
        project_name="tformers",
        workspace="longnguyenha050"  # Thay bằng workspace của bạn
    )

    # get training options
    opt = TrainOptions().parse() #

    # Log hyperparameters lên Comet
    experiment.log_parameters(vars(opt)) #

    # create a dataset
    dataset = dataloader(opt) #
    dataset_size = len(dataset) * opt.batchSize #
    print('training images = %d' % dataset_size) #

    # create a model
    model = create_model(opt) #

    # create a visualizer
    visualizer = Visualizer(opt) #

    # training flag
    keep_training = True #
    max_iteration = opt.niter + opt.niter_decay #
    epoch = 0 #
    total_iteration = opt.iter_count #

    # training process
    while keep_training: #
        epoch_start_time = time.time() #
        epoch += 1 #
        print('\n Training epoch: %d' % epoch) #

        for i, data in enumerate(dataset): #
            iter_start_time = time.time() #
            total_iteration += 1 #
            model.set_input(data) #
            model.optimize_parameters() #

            # Log losses lên Comet
            losses = model.get_current_errors()
            for k, v in losses.items():
                experiment.log_metric(k, v, step=total_iteration)

            # display images on visdom and save images

            if total_iteration % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch)

                # Log ảnh lên Comet
                for name, img_array in visuals.items():
                    # Nếu ảnh là grayscale (H, W), chuyển sang RGB
                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.ndim == 3 and img_array.shape[2] == 1:
                        img_array = np.repeat(img_array, 3, axis=2)

                    # Convert sang PIL rồi log
                    pil_image = Image.fromarray(img_array.astype(np.uint8))
                    experiment.log_image(image_data=pil_image, name=f"{name}_epoch{epoch}_step{total_iteration}")


            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)

            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')

            # save the model every <save_iters_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0:
                print('saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break

        model.update_learning_rate()

    print('\nEnd training')
    experiment.end()
