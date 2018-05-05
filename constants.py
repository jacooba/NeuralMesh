#note, to make fully-connected emulatin realizable, we need:
     #1) no residual, 2) reset energies, 3) at least one time step, 4) dont normalize weights, 5) bias on input
#(weights 0 is how this happens)

#ways energy can escape system:
    #energy used by a negative weight (decreases energy of other neuron)
    #hard deplete to -1 loses nergy to "friction", since it transfers x/(x+eps) energy, instead of 1.. but dont know how to do this
    #energy clipped by 1
#ways energy can get in system:
    #intial or residual intput, bias on the residual (if set)
    #reset to 0 if set
    #energy clipped by -1

import os

MODEL_NAME = "model"

MODEL_DIR = os.path.join(os.getcwd(),"TFModel")
SUMMARY_DIR = os.path.join(os.getcwd(),"Summaries")

TRAIN_DATA_FILE_PATH, TRAIN_LABEL_FILE_PATH = ("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
TEST_DATA_FILE_PATH, TEST_LABEL_FILE_PATH = ("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")


ALL_MODEL_FILES = os.path.join(MODEL_DIR,"*")
ALL_SUMMARY_FILES = os.path.join(SUMMARY_DIR,"*")
MODEL_PATH_PREFIX = os.path.join(MODEL_DIR,MODEL_NAME) #tensorflow will add something to the end of this path 

WINDOW_SZ = 10
IMG_VEC_SZ = 784 #number of pixel values (greyscale). Pixels will be in flattened array.

MESH_ROWS = 30
MESH_COLS = 40

LEARN_RATE = 0.001
TRAINABLE_INIT_STATE = False

#right now, fixed batch size, for GD
NUM_TRAIN_IMGS = 10000 #there are 60,000 total... although keras kills processes when do 60,000?
NUM_TEST_IMGS = 10000 #there are 10,000. MUST DIVIDE the batch size evenly! (we need to use up a whole batch)
BATCH_SZ = NUM_TRAIN_IMGS

NUM_EPOCHS = 5 #15
SUMMARY_FREQ = 2
SAVE_FREQ = 10


USE_INPUT_AS_RESIDUAL = False #whether the window should keep looking at the orinigal image seen.
RESIDUAL_VEC = [0.0 for _ in range(IMG_VEC_SZ)] #if ^^^ is false, the inputs seen by the mesh for all the windows after the first

MAX_TO_0_EACH_STEP = True
NORMALIZE_WEIGHTS = False
CLIP_NEURONS_TO_NEG1_AND_1 = False
BIAS_ON_INPUT = True
DEPLETE_TO_NEG_1 = False 
#HARD_DEPLETE_TO_NEG_1 = False #...dont know how to do this...: if above is set, this will force depelete to actually go all the way to -1, but will lose some energy to friction (see note at top)
assert not (DEPLETE_TO_NEG_1 and (MAX_TO_0_EACH_STEP or not NORMALIZE_WEIGHTS)) #it cannot be the case that you deplete to neg1 and yet are maxing to 0 or not norming weights

ABS_APPROX = True #whether to approx absolute value to a more differentiable approximation: abs val: sqrt(square(x)+EPS^2) - EPS
EPS = 0.0001 #the eps used in computing the differentiabl approx of functions (abs and division)
#^bench marking seems to be pretty sensitive to the above.