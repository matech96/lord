default_config = dict(
	img_shape=(64, 64, 3),

	content_dim=32,
	identity_dim=512,

	n_adain_layers=4,
	adain_dim=256,

	batch_size=64,
	n_total_iterations=1000000,
	n_checkpoint_iterations=10000,
	n_log_iterations=500
)
