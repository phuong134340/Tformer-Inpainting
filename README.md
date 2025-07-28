# T-former: An Efficient Transformer for Image Inpainting (MM 2022) 
This is the code for ACM multimedia 2022 “T-former: An Efficient Transformer for Image Inpainting”

# download dataset

# # CelebA:
test: https://drive.google.com/drive/folders/1s0YvrHsUQZtwJ-3X_73U13QElRRKxjia?usp=drive_link

# set up environment

pip install -r requirement.txt

# set up visdom

apt update

apt install nodejs npm -y

npm install -g localtunnel

# open visdom

python3 start_visdom.py

- sau khi chạy đoạn code trên, mở đường link sau khi được tạo, trong trang web được tạo, nhập vào IP của server đang chạy.

# train:

python3 train.py --no_flip --no_rotation --no_augment --img_file your_data --lr 1e-4

# fine_tune:

python3 train.py --no_flip --no_rotation --no_augment --img_file your_data --lr 1e-5 --continue_train

# test:

python3 test.py --batchSize 1 --img_file your_data your_data

# cancel visdom

visdom_proc.terminate()
lt_proc.terminate()

## Citation

If you are interested in this work, please consider citing:

    @inproceedings{tformer_image_inpainting,
      author = {Deng, Ye and Hui, Siqi and Zhou, Sanping and Meng, Deyu and Wang, Jinjun},
      title = {T-former: An Efficient Transformer for Image Inpainting},
      year = {2022},
      isbn = {9781450392037},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      doi = {10.1145/3503161.3548446},
      pages = {6559–6568},
      numpages = {10},
      series = {MM '22}

}
