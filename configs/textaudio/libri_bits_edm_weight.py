from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()
    
    cfg.framework = 'continuous_score'
    cfg.experiment = 'textaudio_pilot'
    cfg.device = 'cuda'

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = 'libri'
    cfg.data.root = 'datasets/libri'
    cfg.data.text_tokenizer = 'o200k_base'
    cfg.data.speech_tokenizer = 'stabilityai/stable-codec-speech-16k'
    cfg.data.speech_tokenizer_bottleneck = '1x46656_400bps'
    cfg.data.speech_tokenizer_bottleneck_dims = None

    cfg.data.representation = 'binary'
    cfg.data.binarization = 'raw_binary'
    cfg.data.token_space = 'tokenizer_id'

    cfg.data.text_seq_len = 168
    cfg.data.speaker_seq_len = 32
    cfg.data.speech_seq_len = 800
    cfg.data.seq_len_tokens = 1000
    cfg.data.bits_per_token = 18
    cfg.data.sequence_len = 1000*18
    cfg.data.sample_rate = 16000

    cfg.data.text_vocab_size = 200019
    cfg.data.speaker_vocab_size = 4096
    cfg.data.speech_vocab_size = 46656
    cfg.data.speech_offset = 204115
    cfg.data.vocab_size = 2
    cfg.data.vocab_size_base = 250773
    cfg.data.channels = 1
    cfg.data.flatten_order = 'flatten'

    cfg.data.num_workers = 12
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True
    
    cfg.data.partition = 'clean' # Test-only

    # ------------------------------------------------------------------
    # Unconditional benchmark setting
    # ------------------------------------------------------------------
    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = True
    cfg.cond.continuation_prefix = 75
    cfg.cond.unconditional_rate = 0.25
    cfg.cond.texttospeech_rate = 0.25
    cfg.cond.speechtotext_rate = 0.25
    cfg.cond.continuation_rate = 0.25
    cfg.cond.downstream = True

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True
    cfg.model.patch_size = 18

    cfg.model.embed_dim = 768
    cfg.model.dim_ff = 3072
    cfg.model.n_blocks = 12
    cfg.model.n_heads = 12

    cfg.model.head_type = "optimal_skip_mlp"
    cfg.model.out_dim = 1
    cfg.model.head_hidden = 128
    cfg.model.head_embed_dim = 64

    cfg.model.n_pos_features = 1
    cfg.model.dropout = 0.1
    cfg.model.content_dim_discrete = 64
    cfg.model.content_dim_continuous = 64

    cfg.model.head_use_cross_attn = True
    cfg.model.head_use_local_mixer = True
    cfg.model.head_use_self_attn = False
    cfg.model.head_variant = "single"
    cfg.model.head_kernel = 3
    cfg.model.head_dilation = 1

    cfg.model.use_rope_trunk = True
    cfg.model.rope_base = 10_000.0
    cfg.model.abs_pos_mode = "local_only"
    cfg.model.n_fourier_global = 32
    cfg.model.n_fourier_local = 8
    cfg.model.use_adaln = True
    cfg.model.rpb_max_distance = 1
    cfg.model.use_swiglu = True
    cfg.model.scale_by_sigma = False

    cfg.model.continuous_logit_scaling = "matched_filter_residual"
    cfg.model.matched_filter_center = 0.5
    cfg.model.matched_filter_scale = 1.0
    cfg.model.matched_filter_clip = 30.0    


    # ------------------------------------------------------------------
    # Continuous diffusion
    # ------------------------------------------------------------------
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.002
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.diffusion.continuous.p_mean = -1.2
    cfg.diffusion.continuous.p_std = 1.2


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.deterministic = False
    cfg.train.seed = 42
    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    
    cfg.train.loss_weighting = "edm"

    cfg.train.batch_size = 512
    cfg.train.epochs = 200
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_compute = True
    cfg.train.entropy_use_for_sampling = True
    cfg.train.entropy_buffer_size = 800_000
    cfg.train.entropy_num_bins = 128
    cfg.train.entropy_min_per_bin = 100
    cfg.train.entropy_update_every_steps = 2000
    cfg.train.entropy_warmup_steps = 10_000 # TODO: back to 40_000 (check)
    cfg.train.entropy_transition_steps = 3_000
    cfg.train.entropy_gamma_max = 1.0
    cfg.train.entropy_mode = "regularized"
    cfg.train.entropy_regularizer_c = 0.1
    cfg.train.entropy_regularizer_n = 3.0
    cfg.train.entropy_target = "sqrt-rate"
    cfg.train.entropy_plot_every_k_epochs = 5

    cfg.train.checkpointing = config_dict.ConfigDict()
    cfg.train.checkpointing.save_last = True
    cfg.train.checkpointing.save_top_k = 2
    cfg.train.checkpointing.mode = "min"

    cfg.train.checkpointing.interval = config_dict.ConfigDict()
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = 50_000
    cfg.train.checkpointing.interval.keep_last = 0

    cfg.train.checkpointing.resume_interval = config_dict.ConfigDict()
    cfg.train.checkpointing.resume_interval.enabled = True
    cfg.train.checkpointing.resume_interval.every_steps = 5_000

    cfg.train.sanity = config_dict.ConfigDict()
    cfg.train.sanity.enabled = True
    cfg.train.sanity.run_epoch = -1

    cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.enabled = False
    cfg.train.generation.splits = ["val"]
    cfg.train.generation.every_epochs = 5
    cfg.train.generation.num_samples = 64
    cfg.train.generation.num_sampling_steps = 128
    cfg.train.generation.samplers = ["ddim_entropic"]
    cfg.train.generation.terminal_sigmas = [0.08]
    cfg.train.generation.entropic_blend_alpha = 0.0
    cfg.train.generation.entropy_ckpt_path = None
    cfg.train.generation.guidance_scales = [0.0]
    cfg.train.generation.micro_batch_size = 64
    cfg.train.generation.sc_refresh_mode = "carry"
    cfg.train.generation.sigma_max = None

    cfg.train.textaudio = config_dict.ConfigDict()
    cfg.train.textaudio.enabled = True
    cfg.train.textaudio.run_on_sanity = True
    cfg.train.textaudio.every_epochs = 10
    cfg.train.textaudio.split = 'val'
    cfg.train.textaudio.num_samples = 64
    cfg.train.textaudio.whisper_model = 'openai/whisper-medium'


    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 3e-4
    cfg.optim.weight_decay = 0.01
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.99
    cfg.optim.eps = 1e-8
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "cosine_decay"
    cfg.optim.total_steps = 140_000
    cfg.optim.warmup = 2_500

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=001000000.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/evaluation_frontier_step1M"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/evaluation_frontier_step1M/samples"
    cfg.evaluation.results_csv = f"runs/{cfg.experiment}/evaluation_frontier_step1M/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"runs/{cfg.experiment}/evaluation_frontier_step1M/shared_text_cache"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 128
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"

    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = True
    cfg.evaluation.compile.warmup_steps = 8

    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = [8, 16, 32, 64, 128, 256, 512]
    cfg.evaluation.sampling_sweep.specs = [
        config_dict.ConfigDict({
            "sampler_name": "ddim_entropic",
            "sc_refresh_modes": ["carry"],
            "target_nfes": [8, 16, 32, 64, 128],
            "ati_etas": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }),
        config_dict.ConfigDict({
            "sampler_name": "ddim_entropic",
            "sc_refresh_modes": ["carry"],
            "target_nfes": [256, 512, 1024],
            "ati_etas": [0.0, 0.1, 0.2],
        }),
    ]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = True
    cfg.logging.entity = None
    cfg.logging.project = "libri"
    cfg.logging.group = "libri_continuous_raw_binary_bits_trunk768"
    cfg.logging.mode = "online"
    cfg.logging.watch_model = False
    cfg.logging.log_freq = 10
    cfg.logging.run_id = None

    cfg.logging.tensorboard = config_dict.ConfigDict()
    cfg.logging.tensorboard.enabled = True
    cfg.logging.tensorboard.log_dir = "auto"
    cfg.logging.tensorboard.scalar_every_steps = 20
    cfg.logging.tensorboard.flush_secs = 30
    cfg.logging.tensorboard.max_queue = 2000
    cfg.logging.tensorboard.sync_to_run_dir = True
    cfg.logging.tensorboard.sync_every_epochs = 1
    cfg.logging.tensorboard.sync_every_steps = 500
    cfg.logging.tensorboard.copy_existing_to_scratch = True
    cfg.logging.tensorboard.fail_silently = True

    return cfg