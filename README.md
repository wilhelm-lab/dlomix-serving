# dlomix-serving

## Accessing a public server
### curl
Here is an example http request using only curl sending a POST request to with a json body.

```
curl -X POST "http://eubic2023.external.msaid.io:8501/v2/models/Prosit_2019_intensity_ensemble/infer" \
 --data-raw '
{
  "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
  "inputs": [
    {"name": "peptides_in_str:0",         "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
    {"name": "collision_energy_in:0",     "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
    {"name": "precursor_charge_in_int:0", "shape": [2,1], "datatype": "INT32", "data": [1,2]}
  ]
}
'
```


### Python
See the examples in the corresponding [documentation folder](docs/Python/)

### R
TODO


## Hosting your own server

### Dependencies
dlomix-serving depends on [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html). 

You can find an ansible script that installs all dependencies [here](docs/server/).

### How to run it
After installing the dependencies you can pull the docker image and run it with. 
```
docker run \
    --gpus all \
    --shm-size 2G \
    -p 8500:8500 \
    -p 8501:8501 \
    -d \
    ghcr.io/wilhelm-lab/dlomix-serving:latest
```

## Adding your own model

### Set up a development server

1. Install dependencies ([Ansible script](docs/server/))
2. (Suggested) Install [docker-compose](https://docs.docker.com/desktop/install/linux-install/)
3. Clone the repo
4. Download existing models with `./getModels.sh`
5. Update `.env` with your user- and group-id to avoid file permission issues 
6. Start the server with `docker-compose up`

### Import model files
This step depends on what framework you used to train your model.
For detailed instructions in what format your model needs to be provided you can check out this [documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)

You can find examples for [TensorFlow](models/Prosit/Prosit_2019_intensity/), [PyTorch](models/AlphaPept/AlphaPept_ms2_generic/) and [XGBoost](models/ms2pip/model_20210416_HCD2021_Y/) in our model repository. The model files themselves need to be downloaded from Zenodo.

### Create pre- and post-processing steps
Triton supports models written in pure python. If your model requires pre- and/or post-processing you can implement this as a "standalone" model in python.

There are numerous examples in this repository. One with low complexity you can find [here](models/AlphaPept/AlphaPept_Preprocess_charge/).


### Create an ensemble model to connect everything
Ensemble models don't have any code themselves they just manage moving tensors between other models. This is perfect for combining your potentially various pre- and post-processing steps with your main model to create one single model/workflow.
