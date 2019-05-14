default_config = dict(
	n_adain_layers=4,
	adain_dim=256,

	train=dict(
		batch_size=64,
		n_epochs=100000,
		n_epochs_per_decay=500,
		n_epochs_per_checkpoint=50
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=500,
		n_epochs_per_decay=50,
		n_epochs_per_checkpoint=5,

	)
)
