## Prerequesite task

Outputs:

- Testing `my_dataset.py`:
```bash
(fluffy) ~/T/prerequesite-task ❯❯❯ python my_dataset.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar10/cifar-10-python.tar.gz
100.0%Extracting data/cifar10/cifar-10-python.tar.gz to data/cifar10
Files already downloaded and verified

--- CIFAR10 ---
Train set length = 50000
Test set length = 10000
```

- Testing `my_model.py`:
```bash
(fluffy) ~/T/prerequesite-task ❯❯❯ python my_model.py
--- Model info ---
Network(
  (layers): ModuleList(
    (0): ConvBnRelu(
      (conv_bn_relu): Sequential(
        (0): Conv2d(3, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (1): Cell(
      (vertex_op): ModuleList(
        (0): None
        (1): Conv1x1BnRelu(
          (conv1x1): ConvBnRelu(
            (conv_bn_relu): Sequential(
              (0): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
          )
        )
        (2): Conv3x3BnRelu(
          (conv3x3): ConvBnRelu(
            (conv_bn_relu): Sequential(
              (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
          )
        )
        (3): Conv3x3BnRelu(
          (conv3x3): ConvBnRelu(
            (conv_bn_relu): Sequential(
              (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
          )
        )
        (4): Conv3x3BnRelu(
          (conv3x3): ConvBnRelu(
            (conv_bn_relu): Sequential(
              (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
          )
        )
        (5): MaxPool3x3(
          (maxpool): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        )
      )
      (input_op): ModuleList(
        (0): None
        (1): ConvBnRelu(
          (conv_bn_relu): Sequential(
            (0): Conv2d(12, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (2): ConvBnRelu(
          (conv_bn_relu): Sequential(
            (0): Conv2d(12, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (3): ConvBnRelu(
          (conv_bn_relu): Sequential(
            (0): Conv2d(12, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (4): None
        (5): ConvBnRelu(
          (conv_bn_relu): Sequential(
            (0): Conv2d(12, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (6): None
      )
    )
  )
  (classifier): Linear(in_features=12, out_features=10, bias=True)
)
```

- Prepare model using `pyklopp`:
```bash
(fluffy) ~/T/prerequesite-task ❯❯❯ pyklopp init my_model --save 'test/my_model.pth'
Added /home/arsal/Temp/task to path
Loaded modules: []
Configuration:
{
  "argument_model": "my_model",
  "get_model": "get_model",
  "global_unique_id": "d8840de7-c3b1-4171-a7d6-34606ec66ed3",
  "gpus_exclude": [],
  "hostname": "munir",
  "loaded_modules": [],
  "model_persistence_name": "my_model.pth",
  "pyklopp_version": "0.2.2",
  "python_cwd": "/home/arsal/Temp/task",
  "python_seed_initial": null,
  "python_seed_local": 742,
  "python_seed_random_lower_bound": 0,
  "python_seed_random_upper_bound": 10000,
  "save_path_base": "test",
  "time_config_end": 1587915313.9691875,
  "time_config_start": 1587915313.9677074
}
Saving to "test/my_model.pth"
Writing configuration to "test/config.json"
Final configuration:
{
  "argument_model": "my_model",
  "get_model": "get_model",
  "global_unique_id": "d8840de7-c3b1-4171-a7d6-34606ec66ed3",
  "gpus_exclude": [],
  "hostname": "munir",
  "loaded_modules": [],
  "model_persistence_name": "my_model.pth",
  "model_pythonic_type": "<class 'model.Network'>",
  "model_trainable_parameters": 922,
  "pyklopp_version": "0.2.2",
  "python_cwd": "/home/arsal/Temp/task",
  "python_seed_initial": null,
  "python_seed_local": 742,
  "python_seed_random_lower_bound": 0,
  "python_seed_random_upper_bound": 10000,
  "save_path_base": "test",
  "time_config_end": 1587915313.9691875,
  "time_config_start": 1587915313.9677074,
  "time_model_init_end": 1587915313.984467,
  "time_model_init_start": 1587915313.9757226,
  "time_model_save_end": 1587915314.0314097,
  "time_model_save_start": 1587915313.984843
}
Done.
```

- Train model using `pyklopp`:
```bash
(fluffy) ~/T/prerequesite-task ❯❯❯ pyklopp train test/my_model.pth my_dataset.get_dataset_train --save='test/trained.pth' --config='{"dataset_root":"data/cifar10"}'
Added /home/arsal/Temp/task to path
Loading dataset.
Files already downloaded and verified
Configuration:
{
  "argument_dataset": "my_dataset.get_dataset_train",
  "batch_size": 100,
  "config_key": "training",
  "config_persistence_name": "config.json",
  "dataset": "CIFAR10",
  "dataset_class": "get_dataset_train",
  "dataset_root": "data/cifar10",
  "device": "cpu",
  "get_dataset_test": null,
  "get_dataset_transformation": "pyklopp.defaults.get_transform",
  "get_loss": "pyklopp.defaults.get_loss",
  "get_optimizer": "pyklopp.defaults.get_optimizer",
  "global_unique_id": "aca54f1f-2f0d-450d-a354-c5bb5441dab7",
  "gpu_choice": null,
  "gpus_exclude": [],
  "hostname": "munir",
  "learning_rate": 0.01,
  "loaded_modules": [],
  "loss": "CrossEntropyLoss",
  "model_persistence_name": "trained.pth",
  "model_pythonic_type": "<class 'model.Network'>",
  "model_root_path": "test/my_model.pth",
  "num_epochs": 10,
  "optimizer": "SGD",
  "pyklopp_version": "0.2.2",
  "python_cwd": "/home/arsal/Temp/task",
  "python_seed_initial": null,
  "python_seed_local": 3189,
  "python_seed_random_lower_bound": 0,
  "python_seed_random_upper_bound": 10000,
  "save_path_base": "test",
  "time_config_end": 1587915350.6210895,
  "time_config_start": 1587915350.6197908,
  "time_dataset_loading_end": 1587915353.4294558,
  "time_dataset_loading_start": 1587915353.3371782
}
Epoch [1/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [01:54<00:00]
Training - Epoch: 1, accuracy: 0.189, precision: 0.126, recall: 0.189, f1: nan, loss: 2.24
Epoch [2/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:13<00:00]
Training - Epoch: 2, accuracy: 0.197, precision: 0.212, recall: 0.197, f1: 0.119, loss: 2.15
Epoch [3/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:05<00:00]
Training - Epoch: 3, accuracy: 0.223, precision: 0.209, recall: 0.223, f1: 0.177, loss: 2.11
Epoch [4/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:01<00:00]
Training - Epoch: 4, accuracy: 0.234, precision: 0.225, recall: 0.234, f1: 0.2, loss: 2.08
Epoch [5/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:01<00:00]
Training - Epoch: 5, accuracy: 0.243, precision: 0.235, recall: 0.243, f1: 0.218, loss: 2.06
Epoch [6/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:00<00:00]
Training - Epoch: 6, accuracy: 0.256, precision: 0.243, recall: 0.256, f1: 0.229, loss: 2.04
Epoch [7/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:02<00:00]
Training - Epoch: 7, accuracy: 0.261, precision: 0.253, recall: 0.261, f1: 0.239, loss: 2.01
Epoch [8/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [01:58<00:00]
Training - Epoch: 8, accuracy: 0.264, precision: 0.257, recall: 0.264, f1: 0.243, loss: 2.0
Epoch [9/10]: [500/500] 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [02:01<00:00]
Training - Epoch: 9, accuracy: 0.276, precision: 0.27, recall: 0.276, f1: 0.264, loss: 1.98
Epoch [10/10]: [500/500] 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ [01:59<00:00]
Training - Epoch: 10, accuracy: 0.277, precision: 0.269, recall: 0.277, f1: 0.264, loss: 1.96
Saving to "test/trained.pth"
Writing configuration to "test/config.json"
Final configuration:
{
  "argument_dataset": "my_dataset.get_dataset_train",
  "batch_size": 100,
  "config_key": "training",
  "config_persistence_name": "config.json",
  "dataset": "CIFAR10",
  "dataset_class": "get_dataset_train",
  "dataset_root": "data/cifar10",
  "device": "cpu",
  "get_dataset_test": null,
  "get_dataset_transformation": "pyklopp.defaults.get_transform",
  "get_loss": "pyklopp.defaults.get_loss",
  "get_optimizer": "pyklopp.defaults.get_optimizer",
  "global_unique_id": "aca54f1f-2f0d-450d-a354-c5bb5441dab7",
  "gpu_choice": null,
  "gpus_exclude": [],
  "hostname": "munir",
  "learning_rate": 0.01,
  "loaded_modules": [],
  "loss": "CrossEntropyLoss",
  "model_persistence_name": "trained.pth",
  "model_persistence_path": "test/trained.pth",
  "model_pythonic_type": "<class 'model.Network'>",
  "model_root_path": "test/my_model.pth",
  "num_epochs": 10,
  "optimizer": "SGD",
  "pyklopp_version": "0.2.2",
  "python_cwd": "/home/arsal/Temp/task",
  "python_seed_initial": null,
  "python_seed_local": 3189,
  "python_seed_random_lower_bound": 0,
  "python_seed_random_upper_bound": 10000,
  "save_path_base": "test",
  "time_config_end": 1587915350.6210895,
  "time_config_start": 1587915350.6197908,
  "time_dataset_loading_end": 1587915353.4294558,
  "time_dataset_loading_start": 1587915353.3371782,
  "time_model_save_end": 1587917045.7969337,
  "time_model_save_start": 1587917045.7107203,
  "time_model_training_end": 1587917045.7102537,
  "time_model_training_start": 1587915353.4882839,
  "training_accuracy": [
    0.1889,
    0.19662,
    0.22256,
    0.23384,
    0.24326,
    0.25564,
    0.26102,
    0.26434,
    0.27582,
    0.27744
  ],
  "training_f1": [
    NaN,
    0.11914369587162779,
    0.17742115756684868,
    0.20008730510651246,
    0.2177382665027765,
    0.2287974156556857,
    0.23880309300096822,
    0.2433720209716416,
    0.26383808277939264,
    0.26365518698324186
  ],
  "training_loss": [
    2.237966069698334,
    2.1535606780052183,
    2.1070019674301146,
    2.0799231860637666,
    2.057315809249878,
    2.036273601293564,
    2.013588874578476,
    1.9954898543357849,
    1.9818970718383788,
    1.964366902589798
  ],
  "training_precision": [
    0.1255039083443987,
    0.21167941742611734,
    0.2090338674183369,
    0.22492634863222588,
    0.2352614257135884,
    0.24347474398222763,
    0.252805364318726,
    0.25705729140461414,
    0.26988895857446893,
    0.26944400967081483
  ],
  "training_recall": [
    0.1889,
    0.19662000000000002,
    0.22256,
    0.23384,
    0.24325999999999998,
    0.25564000000000003,
    0.26102,
    0.26433999999999996,
    0.27582,
    0.27744
  ]
}
Done.
```

- Evaluate model using `pyklopp`:
```bash
(fluffy) ~/T/prerequesite-task ❯❯❯ pyklopp eval test/trained.pth my_dataset.get_dataset_test --config='{"dataset_root":"data/cifar10"}'
Added "/home/arsal/Temp/task" to path.
Loading dataset.
Files already downloaded and verified
Configuration:
{
  "argument_dataset": "my_dataset.get_dataset_test",
  "batch_size": 100,
  "config_key": "evaluation",
  "config_persistence_name": "config.json",
  "dataset": "CIFAR10",
  "dataset_class": "get_dataset_test",
  "dataset_root": "data/cifar10",
  "device": "cpu",
  "get_dataset_transformation": "pyklopp.defaults.get_transform",
  "get_loss": "pyklopp.defaults.get_loss",
  "global_unique_id": "270d9b9b-d15f-434a-a8e6-0b78944893a9",
  "gpu_choice": null,
  "gpus_exclude": [],
  "hostname": "munir",
  "loaded_modules": [],
  "loss": "CrossEntropyLoss",
  "model_path": "test/trained.pth",
  "model_pythonic_type": "<class 'model.Network'>",
  "pyklopp_version": "0.2.2",
  "python_cwd": "/home/arsal/Temp/task",
  "python_seed_initial": null,
  "python_seed_local": 7025,
  "python_seed_random_lower_bound": 0,
  "python_seed_random_upper_bound": 10000,
  "save_path_base": null,
  "time_config_start": 1587918766.2113948,
  "time_dataset_loading_end": 1587918769.009696,
  "time_dataset_loading_start": 1587918768.902238
}
Final configuration:
{
  "argument_dataset": "my_dataset.get_dataset_test",
  "batch_size": 100,
  "config_key": "evaluation",
  "config_persistence_name": "config.json",
  "dataset": "CIFAR10",
  "dataset_class": "get_dataset_test",
  "dataset_root": "data/cifar10",
  "device": "cpu",
  "evaluation_accuracy": 0.28,
  "evaluation_f1": 0.2626425549854763,
  "evaluation_loss": 1.9525681722164154,
  "evaluation_precision": 0.28445570210054244,
  "evaluation_recall": 0.27999999999999997,
  "get_dataset_transformation": "pyklopp.defaults.get_transform",
  "get_loss": "pyklopp.defaults.get_loss",
  "global_unique_id": "270d9b9b-d15f-434a-a8e6-0b78944893a9",
  "gpu_choice": null,
  "gpus_exclude": [],
  "hostname": "munir",
  "loaded_modules": [],
  "loss": "CrossEntropyLoss",
  "model_path": "test/trained.pth",
  "model_pythonic_type": "<class 'model.Network'>",
  "pyklopp_version": "0.2.2",
  "python_cwd": "/home/arsal/Temp/task",
  "python_seed_initial": null,
  "python_seed_local": 7025,
  "python_seed_random_lower_bound": 0,
  "python_seed_random_upper_bound": 10000,
  "save_path_base": null,
  "time_config_start": 1587918766.2113948,
  "time_dataset_loading_end": 1587918769.009696,
  "time_dataset_loading_start": 1587918768.902238,
  "time_model_evaluation_end": 1587918777.2224846,
  "time_model_evaluation_start": 1587918769.1707256
}
Done.
```

Reference:
1. https://github.com/innvariant/pyklopp
2. https://github.com/romulus0914/NASBench-PyTorch/