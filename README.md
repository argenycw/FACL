# Fourier Amplitude and Correlation Loss (FACL)
This is the official implementation for the paper **Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting** submitted to NeurIPS 2024.

## **Environment**
Create a new conda environment:

```bash
conda create -n facl python=3.9
conda activate facl
```

Install related packages
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Earthformer Installation (Optional)
Since we do not implement Earthformer natively, our code requires a trimmed version of the [Earthformer package](https://drive.google.com/file/d/1ooHpCoFWYPF6xk-NrgrHzfKAzY2Aq-KO/view?usp=sharing). After downloading and unzipping the `earthformer-minimal` folder to the same directory of this README, `cd` into `earthformer-minimal` and run:

```bash
python3 -m pip install pytorch_lightning==1.6.4
python3 -m pip install xarray netcdf4 opencv-python
python3 -m pip install -U -e . --no-build-isolation
```



## **Data Preparation**

### (Stochastic) Moving-MNIST
Both Moving-MNIST and Stochastic Moving-MNIST share the same data source MNIST. To prepare the MNIST training sets and the test set, download [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cwyan_connect_ust_hk/EpgKw7a0xVRNrEFVaUMMfBMBOoOBZ29ZIMvEa1g-xu891w?e=uiHsNp).

The foloer `moving_mnist` includes three files in total:
- `train-images-idx3-ubyte.gz` is the data source of MNIST.
- `mnist_test_seq.npy` is the test set for Moving-MNIST. We will not use it here.
- `smnist_gtest_seq.npy` is our pre-sampled test set for evaluating Stochastic Moving-MNIST.

After downloading the files, place them inside `data/moving_mnist/`

### SEVIR

To download the SEVIR dataset, refer to https://github.com/amazon-science/earth-forecasting-transformer \
In particular, make sure the files are placed inside `data/sevir/` with the following structure:
```
data/
├─ sevir/
│   ├─ data/
│   │   └─ vil/
|   |      ├─ 2017/
|   |      ├─ 2018/
|   |      └─ 2019/
│   └─ CATALOG.csv
├─ ...
```

### HKO-7

To download the HKO-7 dataset, refer to https://github.com/sxjscience/HKO-7.
After obtaining the dataset, place the files inside `data/hko-7` with the following structure:
```
data/
├─ hko-7/
│   ├─ hko_data/
│   │   └─ mask_dat.npz
│   ├─ radarPNG/
│   │   ├─ 2009/
│   │   └─ ...
│   ├─ radarPNG_mask/
│   │   ├─ 2009/
│   │   └─ ...
│   └─ samplers/
├─ ...
```
The filter files in `samplers/` can be downloaded in the same [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cwyan_connect_ust_hk/EpgKw7a0xVRNrEFVaUMMfBMBOoOBZ29ZIMvEa1g-xu891w?e=uiHsNp) link inside `hko-7/samplers`.


### MeteoNet

Download the raw data from https://meteonet.umr-cnrm.fr/dataset/data/SE/radar/reflectivity_old_product/

Unzip the gzip file into the following structure:
```
data/
├─ meteonet/
│   ├─ 2016/
│   │   ├─ reflectivity-old-NW-2016-01/
│   │   │   ├─ reflectivity_old_NW_2016_01.1.npz
│   │   │   └─ ...
│   │   ├─ reflectivity-old-NW-2016-02/
│   │   └─ ...
│   ├─ 2017/
│   │   └─ ...
│   ├─ 2018/
│   │   └─ ...
├─ ...
```

## **Training**

The training script to be called depends on the dataset. Each dataset should have one training script associated, and each training script supports all the models defined.
- Stochastic Moving-MNIST: `train_smmnist.py` (loader in epochs)
- SEVIR: `train_sevir.py` (loader in epochs)
- MeteoNet: `train_meteo.py` (loader in epochs)
- HKO-7: `train_hko7.py` (loader in steps)

To train a baseline ConvLSTM model on Stochastic Moving-MNIST with MSE loss:
```
python train_smmnist.py -m CONVLSTM_MMNIST --loss mse
```
where the argument `-m` specifies the global variables pre-defined in `config.py` and `--loss` has to be one of:
- mae
- mse
- facl (hyphenated with the value of $\alpha$)


To train a PredRNN model on SEVIR with FACL loss and $\alpha=0.1$:
```
python train_sevir.py -m PREDRNN_SEVIR_SIGMOID --loss facl-0.1
```
Note that the suffix `-0.1` is mandatory and we must use models with `_SIGMOID` when applying FACL. 

For training on HKO-7, the argument is slightly different. Apart from the previous arguments, we also need to specify two positional arguments: the sampled days for training and test, where the pkl files can be found inside `data/HKO-7/samplers`
```
python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m SIMVP_HKO7_SIGMOID --loss facl-0.1
```

## **Evaluation**

There is only a file responsible for evaluation: `eval.py`.

Again, we need to specify the global configuration used during training to reconstruct the dataset and models for evaluation. In particular, there are three major parameters:
- `-f` (checkpt path): The full/relative path to the model parameters
- `-d` / `--dataset`: The global dataset config defined in `config.py`
    - They are mostly in the format `<Dataset-name>`\_`<seq_len>`\_`<out_len>` (e.g. `HKO7_5_20`, `SEVIR_13_12`, etc.)
- `-m` / `--model`: The global model config defined in `config.py`
    - They are all in the format `<Model-name>`\_`<Dataset-name>`\_(optional)`SIGMOID` \
    (e.g. `CONVLSTM_HKO7`, `CONVLSTM_HKO7_SIGMOID`, etc.)


For example, the following is a valid command:
```
python eval.py -d SEVIR_13_12 -m SIMVP_SEVIR -f path/to/checkpoint.pt
```

## **Citation**

If you find our work helpful, please cite the following:
```
@inproceedings{yan2024fourier,
  title={Fourier Amplitude and Correlation Loss: Beyond
Using L2 Loss for Skillful Precipitation Nowcasting},
  author={Yan, Chiu-Wai and Foo, Shi Quan and Trinh, Van Hoan and Yeung, Dit-Yan and Wong, Ka-Hing and Wong, Wai-Kin},
  booktitle={NeurIPS},
  year={2024}
}
```