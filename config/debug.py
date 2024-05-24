from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DEVICE = "cuda"
_C.SEED = 0
_C.DATA_TYPE = 'float'  # or double

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.MODEL_DIR = "output/models"
_C.OUTPUT.MODEL_NAME = "model.pth"

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'LSTM' # 'LSTM' or 'LAN'
_C.MODEL.LSTM_HIDDEN = 10
_C.MODEL.LINE_HIDDEN = 16
_C.MODEL.USED_LAYERS = 1


# -----------------------------------------------------------------------------
# DESIGN
# -----------------------------------------------------------------------------
_C.DESIGN = CN()
_C.DESIGN.SUBLEN = 2048 # size of the raw signal's piece
_C.DESIGN.PIECE = 256 # num of pieces drawed from raw signal
_C.DESIGN.M = 100
_C.DESIGN.P = 10
_C.DESIGN.FPIECE = 200 # num of pieces drawed from feature series
_C.DESIGN.FSUBLEN = _C.DESIGN.M + _C.DESIGN.P
_C.DESIGN.PIECE = _C.DESIGN.FPIECE * (_C.DESIGN.M + _C.DESIGN.P)


# -----------------------------------------------------------------------------
# FEATURE
# -----------------------------------------------------------------------------
_C.FEATURE = CN()
_C.FEATURE.NEED_VIEW = True
_C.FEATURE.MAX_LENGTH = 1024000 # max length of the raw signal used to calcuate features
_C.FEATURE.USED_F = ['KV']
_C.FEATURE.USED_THRESHOLD  = ['Z']
_C.FEATURE.CHANNEL_SCORE_MODE = 'sum' # calculate DTW with each channel or get sum of differernt channel
    # 'sum' or 'every'
_C.FEATURE.NORMAL_PATHS = [r'D:\GithubRepos\fault_estimation_master_design\data\datasets\XJTU-SY\Bearing1_1\1.csv']

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NORMAL_PATH = r'D:\GithubRepos\fault_estimation_master_design\data\datasets\XJTU-SY\Bearing1_1\1.csv'
_C.TRAIN.FAULT_PATH = r'D:\GithubRepos\fault_estimation_master_design\data\datasets\XJTU-SY\Bearing1_1\123.csv'
_C.TRAIN.CHECKPOINT_PERIOD = 10
_C.TRAIN.NEED_CHECKPOINT = False
_C.TRAIN.NEED_PLOT_LOSS = True


# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 1
_C.INFERENCE.UNKWON_PATH = ''
_C.INFERENCE.MODEL_PATH = 'output/xjtu_lstm.pth'
_C.INFERENCE.TEST_CONTENT = r'data\datasets\XJTU-SY\Bearing1_1'
_C.INFERENCE.MAE_ratio_threshold = 0.5

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam" # "SGD" or "Adam

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



# -----------------------------------------------------------------------------
# LOG
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.DIR = "./log"
_C.LOG.ITER_INTERVAL = 100
_C.LOG.EPOCH_INTERVAL = 10
_C.LOG.OUTPUT_TO_FILE = False # 是否输出到文件，默认输出到控制台
_C.LOG.PREFIX = "GUI_default" # 输出到文件的命名前缀

# -----------------------------------------------------------------------------
# DRAW
# -----------------------------------------------------------------------------
_C.DRAW = CN()
_C.DRAW.HIST_BIN = 18
_C.DRAW.THRESHOLD_COLORS = ['red','orange','aqua','lime','violet','gold']  # 指定不同的颜色

# -----------------------------------------------------------------------------
# DENOISE
# -----------------------------------------------------------------------------
_C.DENOISE = CN()
_C.DENOISE.NEED = False
_C.DENOISE.METHOD = 'wavelet' # 'smooth' or 'wavelet'
_C.DENOISE.SMOOTH_STEP = 3
_C.DENOISE.WAVELET = 'db4'
_C.DENOISE.LEVEL = 4
_C.DENOISE.SHOW_TYPE = 'original only' # 'denoised only' or 'both' or 'original only'
_C.DENOISE.SHOW_METHOD = 'wavelet' # 'smooth' or 'wavelet'
_C.DENOISE.SHOW_SMOOTH_STEP = 3
_C.DENOISE.SHOW_WAVELET = 'db4'
_C.DENOISE.SHOW_LEVEL = 4