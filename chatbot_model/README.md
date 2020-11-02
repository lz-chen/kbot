


## Troubleshooting

##### Issues wrt tf version 

Latest change was to use tf version 2.3, but have to install [hparams](https://github.com/tensorflow/tensor2tensor) (from `tensor2tensor`) and
 [graph_def_editor]() manually.

Use:
```
# update tensorflow version
pipenv run pip uninstall tensorflow
pipenv run pip install tensorflow-gpu

# install what's necessary for using hparams
pipenv run pip install tensor2tensor

#install what's necessary for using graph_editor
git clone https://github.com/CODAIT/graph_def_editor.git
pip install ./graph_def_editor
```
In the code, replace `import tensorflow as tf` by `import tensorflow._api.v2.compat.v1 as tf
`; and replace `import tf.contrib.graph_editor as ge` by `import graph_def_editor as ge`;
 replace ` from tf.contrib.training import HParams` by `from tensor2tensor.utils.hparam import HParams`
 
 
##### Correctly install CUDNN

Download `cuDNN Library for Linux (x86)` from [here](https://developer.nvidia.com/rdp/cudnn-download)
 and follow the instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux)
 but not all ([modified steps](https://stackoverflow.com/questions/49656725/importerror-libcudnn-so-7-cannot-open-shared-object-file-no-such-file-or-dire))
