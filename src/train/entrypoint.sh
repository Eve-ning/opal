#!/bin/bash

# Run the training script
python3 -m opal.train

# Run Tensorboard to view the results
python3 -m tensorboard.main --logdir=./lightning_logs --bind_all
