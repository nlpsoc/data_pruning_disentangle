# Disentangling the Roles of Representation and Selection in Data Pruning

Here is the code for our ACL 2025 paper "Disentangling the Roles of Representation and Selection in Data Pruning", by Yupei Du, Yingjin Song, Hugh Mee Wong, Daniil Ignatev, Albert Gatt, and Dong Nguyen. 

## Get Started

### Install dependencies

```bash
python3.10 -m venv py310_venv
source py310_venv/bin/activate
pip install -r requirements.txt
# follow: https://trak.readthedocs.io/en/latest/install.html
pip install traker[fast]
```

### Download the dataset

Download the dataset from [Google Drive](https://drive.google.com/file/d/1WS45lero7EFnyPIQIMyHheTmw23TSjNe/view?usp=sharing), 
and extract it to the `data` directory.

### Train models

Our training script uses a YAML configuration file to specify the training parameters. 
You can find example configurations in the `configs` directory.
We also offer a function to construct a YAML configuration file based on the provided parameters, see `construct_train_yaml` in `exp_utils.py`.
```bash
python train.py <YAML_CONFIG>
```

### Collect representations
To collect representations from different training runs, you can use the `collect_grad_reps.py` script.
Similar to the training script, it requires a YAML configuration file to specify the parameters; 
and you can find example configurations in the `configs` directory, 
or construct a YAML configuration file using the `construct_feature_yaml` function in `exp_utils.py`.
```bash
python collect_grad_reps.py <YAML_CONFIG>
```

### Infer selected data instances and retrain models
To infer the selected data instances, you can use the `subset_inference.py` script. 
You can similarly use the `train.py` script to retrain the models on the selected data instances by specifying the `--selected_uid_path` argument.

### Toy example 
Our code for the toy example in Figure 2a can be found at `sampling_toy.ipynb`. 
