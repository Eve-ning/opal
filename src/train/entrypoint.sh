#!/bin/bash

# Run the training script
python3 -m opal.train

# Run Tensorboard to view the results
tensorboard --logdir=/var/lib/opal/lightning_logs --bind_all
