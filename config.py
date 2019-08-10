default_config = dict(
	pose_std=1,
	n_adain_layers=4,
	adain_dim=256,

	train=dict(
		batch_size=64,
		n_epochs=10000
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=500
	)
)
