from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.LSTM_HIDDEN = 10
_C.MODEL.LINE_HIDDEN = 16
_C.MODEL.USED_LAYERS = 1


# -----------------------------------------------------------------------------
# DESIGN
# -----------------------------------------------------------------------------
_C.DESIGN = CN()
_C.DESIGN.SUBLEN = 2048 # size of the raw signal's piece
_C.DESIGN.PIECE = 256 # num of pieces drawed from raw signal
_C.DESIGN.M = 50
_C.DESIGN.P = 5
_C.DESIGN.FPIECE = 100
_C.DESIGN.FSUBLEN = _C.DESIGN.M + _C.DESIGN.P
_C.DESIGN.PIECE = _C.DESIGN.FPIECE * (_C.DESIGN.M + _C.DESIGN.P)


# -----------------------------------------------------------------------------
# FEATURE
# -----------------------------------------------------------------------------
_C.FEATURE = CN()
_C.FEATURE.MAX_LENGTH = 1024000 # max length of the raw signal used to calcuate features
_C.FEATURE.USED_F = ['Mean']

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NORMAL_PATH = ''
_C.DATASETS.FAULT_PATH = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ----------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""

# -----------------------------------------------------------------------------
# LOG
# -----------------------------------------------------------------------------
_C.LOG_DIR = "./log"