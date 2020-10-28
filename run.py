from pytorch_lightning import Trainer
from task import CoolSystem

model = CoolSystem()

# most basic trainer, uses good defaults
trainer = Trainer(gpus=1, precision=16, progress_bar_refresh_rate=5, max_epochs=10)
trainer.fit(model)