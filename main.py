try:
    from noisy_labels import trainer
except ModuleNotFoundError:
    from src.noisy_labels import trainer
trainer.ciao()
