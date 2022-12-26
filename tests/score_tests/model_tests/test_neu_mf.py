import logging

import numpy as np
import pytorch_lightning as pl


def test_neu_mf_net():
    from opal.conf.conf import MODEL_DIR
    from opal.score.collaborative_filtering import NeuMF
    from opal.score.datamodule import ScoreDataModule

    dm = ScoreDataModule(
        ds_yyyy_mm="2022_11", batch_size=256, m_min_support=50, u_min_support=50,
        score_bounds=(7.5e5, 1e6)
    )

    dm.setup()

    net = NeuMF.load_from_checkpoint(
        (MODEL_DIR / "tiny/checkpoints/epoch=2-step=6144.ckpt").as_posix(),
        uid_no=4007,
        mid_no=6189,
        mf_emb_dim=64,
        mlp_emb_dim=64,
        mlp_chn_out=8,
        scaler=dm.scaler_accuracy,
        lr=0.005
    )

    # Set to evaluate mode (no gradient update)
    net.eval()
    trainer = pl.Trainer(
        accelerator='cpu',
        limit_predict_batches=16,
    )

    y_preds = []
    y_trues = []
    for y_pred, y_true in trainer.predict(net, datamodule=dm):
        y_preds.append(y_pred)
        y_trues.append(y_true)
    y_preds = np.stack(y_preds).flatten()
    y_trues = np.stack(y_trues).flatten()

    def get_error(y_preds, y_trues, a, b):
        y_preds = y_preds[(y_trues >= a) & (y_trues < b)]
        y_trues = y_trues[(y_trues >= a) & (y_trues < b)]

        return np.abs(y_preds - y_trues).mean(), ((y_preds - y_trues) ** 2).mean() ** 0.5

    bounds = np.linspace(0.9, 1, 11)
    logging.info(f"(a)-(b)\tMAE\t\tRMSE")
    for a, b in zip(bounds[:-1], bounds[1:]):
        mae, rmse = get_error(y_preds, y_trues, a, b)
        logging.info(f"{int(a * 100)}-{b:.0%}\t{mae:.2%}\t{rmse:.2%}")
