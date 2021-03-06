default_config = dict(
	# content_std=1,
	# content_decay=0.01,

	n_adain_layers=4,
	# adain_enabled=True,
	adain_dim=256,
	# adain_normalize=True,

	perceptual_loss=dict(
		layers=[2, 5, 8, 13, 18],
		weights=[1, 1, 1, 1, 1],
		scales=[64, ]
	),

	train=dict(
		batch_size=64,
		n_epochs=200
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=200
	)
)
