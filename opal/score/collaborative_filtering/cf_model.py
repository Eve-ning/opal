import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from opal.score.collaborative_filtering.cf_dataset import CFDataset
from opal.score.collaborative_filtering.lit_neu_mf_net import LitNeuMFNet

ds = CFDataset()

net = LitNeuMFNet(ds.uid_no, ds.mid_no, 16, 16, 8)

early_stop_callback = EarlyStopping(monitor="val_mae", min_delta=0.00, patience=5, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=1,
    accelerator='gpu',
    callbacks=[early_stop_callback]
)
trainer.fit(net, train_dataloaders=ds.train_dl, val_dataloaders=ds.val_dl)
