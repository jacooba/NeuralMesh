#note, to make fully-connected emulatin realizable, we need:
     #1) no residual, 2) reset energies, 3) at least one time step, 4) dont normalize weights, 5) bias on input
#(weights 0 is how this happens)

#ways energy can escape system:
    #energy used by a negative weight (decreases energy of other neuron)
    #hard deplete to -1 loses nergy to "friction", since it transfers x/(x+eps) energy, instead of 1.. but dont know how to do this
    #energy clipped by 1
#ways energy can get in system:
    #intial or residual intput, bias on the residual (if set)
    #reset to 0 if set. (but really just bringing closer to 0... it took postive E to get that negative...)
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

#10 default 
WINDOW_SZ = 25 #100 #25 #10  size: 1-(97.1,92.6) 2-(97.6,92.9) 3-(97.2,92.6) 4-(97.5,92.8) 8-(97.8,93.0)
#only config that gets better with window size is all set to false!!!
IMG_VEC_SZ = 784 #number of pixel values (greyscale). Pixels will be in flattened array.

#35 is default
MESH_ROWS = 25 #25 #35 #20
MESH_COLS = 25 #25 #35 #20 #dont see much benefit past 20 or 25
#doesn't seem to affect. even 1x1200 had same effect as 40x30

#10x35x35 is assumed default for saved plots, but I left at 25x25x25 since it gets better with window size

LEARN_RATE = 0.001
START_E = 0.0 #the neuron values at the start
TRAINABLE_INIT_STATE = False #learn an additional amount of energy to start each neuron

#right now, fixed batch size, for GD. 10k for both deault
NUM_TRAIN_IMGS = 10000 #250 #2500 #10000 #there are 60,000 total but I consider *10k* to be full data since its plent.
NUM_TEST_IMGS = 10000 #250 #2500 #10000 
#NOTE: interesting that neural mesh (large window size without clip) does better when they both few neurons (especially if little data)
#e.g. mesh of size 1000x4x4 with 250 train and test got (0.376, 0.348) vs benchmark-(0.3, 0.236).. but can't get more than that. takes too long.
#and WITH clip, with medium data, it does better when there are tons of neurons (4x105x105). 87.24 vs 87.05% test accuracy.
STOCHASTIC_GD = True 
if STOCHASTIC_GD:
    #in this case, batch size must divide NUM_TRAIN_IMGS and NUM_TEST_IMGS
    BATCH_SZ = 2500
    #try benchmark but with way larger window size?
else:
    #in this case, NUM_TEST_IMGS, MUST DIVIDE the batch size evenly! (we need to use up a whole batch)
    BATCH_SZ = NUM_TRAIN_IMGS
BENCHMARK_STOCHASTIC = True #wehther benchmark FF model is stochastic (leave false since non-SGD does better here)???
#stochastic actually seems to be doing better? will benchmark in a sec, but:
#with batch_sz = 2500, acc = (97.52,  93.619996) and only 30 epoch
#I have energy plot of above
#... turned out, yes, Mesh is better than FF, at least when it is SGD and FF is GD and 30 epoch
# as for both SGD...
#... ahh shit, SGD is better for both... redoing benchmarks
# SGD better becuae it allows for more steps of same learning rate in same amount of time. (and you dont need to be that confideng of lr sometimes)
# it lets you do more steps or have smaller learning rate in same amount of time
# really just faster
#OK, so with window size = 100 and SGD, Nueral mesh does get better for large neurons (like 25^2)!!
# but I think this was opposite with regular GD (in that NM was better with few neurons and little data)
#... will need to confirm on GD plots... may just be *coincidence*


#30 is default for SGD, 50 is default for GD 
NUM_EPOCHS = 30 #50 #100
SUMMARY_FREQ = 10 #in epochs
SAVE_FREQ = 25


USE_INPUT_AS_RESIDUAL = False #whether the window should keep looking at the orinigal image seen.
RESIDUAL_VEC = [0.0 for _ in range(IMG_VEC_SZ)] #if ^^^ is false, the inputs seen by the mesh for all the windows after the first

#defualt for this these architectural params / booleans is ALL FALSE! (all these were tuned on GD, which was old default before starting the paper)
# ***they were chosen / tuned, however, with 100 epoch and 20x20 and window 10 [10x20x20] (not deault values for those params!)
#all false works wonders like (94, 92)! -> max(97,94)-pic, bias(94,92), maxbias(96,94)
#norm, clip, deplete all together doesnt work (this is what i really want)
#norm and clip doesnt work
#norm and deplete gets like (65, 64)
#clip and deplete (90, 89) -> max+bias, max, bias -> same or worse
#just clip: **(98, 93)**! ->try to boost with bias or max -> not really
#just depete -1: (90, 89)
#just norm: (70,70)
##
## to recap, best were: all false, just clip, max(w/ or w/o bias)
# a tad bit better with clip or max... both can make more like traditional.. but still no bias...
MAX_TO_0_EACH_STEP = False
NORMALIZE_WEIGHTS = False
CLIP_NEURONS_TO_NEG1_AND_1 = False
BIAS_ON_INPUT = False
DEPLETE_TO_NEG_1 = False #allows depleteion down to -1. (not rewuried unless norm weights)
#HARD_DEPLETE_TO_NEG_1 = False #...dont know how to do this...: if above is set, this will force depelete to actually go all the way to -1, but will lose some energy to friction (see note at top)
#actaully fine... assert not (DEPLETE_TO_NEG_1 and (MAX_TO_0_EACH_STEP or not NORMALIZE_WEIGHTS)) #it cannot be the case that you deplete to neg1 and yet are maxing to 0 or not norming weights

ABS_APPROX = True #whether to approx absolute value to a more differentiable approximation: abs val: sqrt(square(x)+EPS^2) - EPS
EPS = 0.0001 #the eps used in computing the differentiabl approx of functions (abs and division)
#^bench marking seems to be pretty sensitive to the above.

#note, in the "clip" case, normalization will be done by scaling, so 1) MANNUAL_SIGMOID_FOR_DISPLAY will be ignore and 2) IMSHOW_NORM_FOR_DISPLAY should probs be false
#unlikely you want both? likely you want one if you arent clippin?
#although, having it clip to [0,1] does produce some nice effects, although just shows small pos energies
MANNUAL_SIGMOID_FOR_DISPLAY = True #wether to use sigmoid to scale for images
IMSHOW_NORM_FOR_DISPLAY = False #wether to (also) use built in to imshow. this can mess up temps between timesteps.
#generally do NEITHER (to look whether values are pos or neg and variation in small pos) or just SIGMOID!!!



##########################
#MORE REMARKS#
##########################
#BENCHMARKING THIS ON MNIST IS HIGHLIGHTS HOW BENCHMARKING THIS IS A BAD IDEA
# the goal is not to game some loss function. We dont know loss function that results in human brain.
# but we need this so that, when we do figure it out, we are ready with the internal represenation!!!