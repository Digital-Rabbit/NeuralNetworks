/***********************************
	Back Propagation Network

	17 Jan 2015	
	B. Laden
	
	Based on work by D.E. Rumelhart, J.L. McClelland
	"Explorations on Parallel Distributed Processing"
	
	Work in progress. Still working on reading input, output, target.
	The network and training routines are in place.
	
	1. Sets a seed for the pseudorandom number generator. This allows the 
	work to be reproducible.

	2. Generates the network.Initialize the network by allocating memory, 
	defining topology, assigning starting weights, and setting input and target patterns

	3. Assign random starting weights. There are more modern techniques that 
	assign weights based on specific probability distributions. That approach has 
	proven to decrease training time. I’d like to investigate further and try out that 
	approach in this app.

	4. Open a file for saving weights. The weights are saved either when the error 
	criteria is achieved or when the maximum epochs is reached.
	
	5. Training proceeds by reading in input, propagating the input through the network,
	comparing output to a target, calculating the error, adjusting weights by using
	the standard backpropagation algorithm. (Might want to change this approach to 
	try resilient backpropagation.

***********************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/***********************************
	Global Declarations
***********************************/

typedef int           Bool;

#define FALSE         0
#define TRUE          1

#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1.0

#define sqr(x)        ((x)*(x))

double testError	  +HUGE_VAL

/* Structure to define one layer in a network */
typedef struct {                       
        int             units;         /* number of units in this layer */
        double*         output;        /* output of ith unit  */
        double*         error;         /* error term of ith unit   */
        double**        weight;        /* connection weights to ith unit      */
        double**        weightSave;    /* saved weights  */
        double**        dWeight;       /* last weight deltas used for momentum  */
} Layer;

/* Structure to define a network with one input, one output, and n hidden layers */
typedef struct {                    
        Layer**       layer;         /* layers in this network  */
        Layer*        inputLayer;   
        Layer*        outputLayer;   
        double        momentum;      
        double        learningRate;       
        double        sigmoidGain;    /*  sigmoid function gain */
        double        totalError;     /*  total error of the network */
} Net;


/***********************************
   Random routines
***********************************/

void seedPseudoRandomGenerator()
{
  srand(4711);
}


int getRandomInteger(int Low, int High)
{
  return rand() % (High-Low+1) + Low;
}      


double getRandomDouble(double Low, double High)
{
  return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}      


/***********************************
	Define network and patterns for
	a specific problem.
***********************************/


#define NUM_LAYERS    		  3  
#define NUM_INPUT             8
#define NUM_OUPUT             8
int                   		units[NUM_LAYERS] = {NUM_INPUT, 16, NUM_OUTPUT};
int	maxPatterns				= 3;
int maxEpochs				= 100;
double errorCriteria        = 0.1;

/* For this initial program, three patterns with 8 inputs each */
double                  thePattern [3][8] = {
                        0, 0, 0, 0, 5, 5, 5, 5,
						0, 0, 5, 5, 0, 0, 5, 5,
						0, 5, 0, 5, 0, 5, 0, 5
                      };

/* For this initial program, output pattern has 8 nodes */                      
double					theTarget [3][8] = {
						1, 0, 0, 0, 0, 0, 0, 0,
						0, 1, 1, 1, 1, 1, 1, 0,
						0, 0, 0, 0, 0, 0, 0, 1
						};


FILE*                 	outputFile;


void normalizeData()
{
  /* TBD: Add normalization. Not using this now  */
}


void initializeApp(Net* net)
{
  /* opening file to save intermediate and final results */
  outputFile = fopen("BP_Results.txt", "w");
}


void cleanUpTasks(Net* net)
{
  /* Add other tasks as needed */
  fclose(outputFile);
}


/***********************************
	Initialize the network by defining topology, assigning starting weights, 
	and setting input and target patterns
***********************************/


void generateNetwork(Net* net)
{
  int l,i;

  net->layer = (Layer**) calloc(NUM_LAYERS, sizeof(Layer*));
   
  for (l=0; l<NUM_LAYERS; l++) {
    net->layer[l] = (Layer*) malloc(sizeof(Layer));
      
    net->layer[l]->units      = units[l];
    net->layer[l]->output     = (double*)  calloc(units[l]+1, sizeof(double));
    net->layer[l]->error      = (double*)  calloc(units[l]+1, sizeof(double));
    net->layer[l]->weight     = (double**) calloc(units[l]+1, sizeof(double*));
    net->layer[l]->weightSave = (double**) calloc(units[l]+1, sizeof(double*));
    net->layer[l]->dWeight    = (double**) calloc(units[l]+1, sizeof(double*));
    net->layer[l]->output[0]  = BIAS;
      
    if (l != 0) { /* input layer does not have weights coming to it */
      for (i=1; i<=units[l]; i++) {
        net->layer[l]->weight[i]     = (double*) calloc(units[l-1]+1, sizeof(double));
        net->layer[l]->weightSave[i] = (double*) calloc(units[l-1]+1, sizeof(double));
        net->layer[l]->dWeight[i]    = (double*) calloc(units[l-1]+1, sizeof(double));
      }
    }
  }
  net->inputLayer  = net->layer[0];
  net->outputLayer = net->layer[NUM_LAYERS - 1];
  net->momentum        = 0.9;
  net->learningRate    = 0.25;
  net->sigmoidGain     = 1;
}


void assignRandomWeights(Net* net)
{
  int l,i,j;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=net->layer[l]->units; i++) {
      for (j=0; j<=net->layer[l-1]->units; j++) {
        net->layer[l]->weight[i][j] = getRandomDouble(-0.5, 0.5);
      }
    }
  }
}


  
void setInput(Net* net, double* input)
{
  int i;
   
  for (i=1; i<=net->inputLayer->units; i++) {
    net->inputLayer->output[i] = input[i-1];
  }
}



void getOutput(Net* net, double* output)
{
  int i;
   
  for (i=1; i<=net->outputLayer->units; i++) {
    output[i-1] = net->outputLayer->output[i];
  }
}


/***********************************
	Functions for saving and restoring weights in the event you need to 
	stop the training.
***********************************/


void saveWeights(Net* net)
{
  int l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=net->layer[l]->units; i++) {
      for (j=0; j<=net->layer[l-1]->units; j++) {
        net->layer[l]->weightSave[i][j] = net->layer[l]->weight[i][j];
      }
    }
  }
}


void restoreWeights(Net* net)
{
  int l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=net->layer[l]->units; i++) {
      for (j=0; j<=net->layer[l-1]->units; j++) {
        net->layer[l]->weight[i][j] = net->layer[l]->weightSave[i][j];
      }
    }
  }
}


/***********************************
	Propogating values through the 
	layers of the network.
***********************************/


void propagateThroughLayer(Net* net, Layer* lower, Layer* upper)
{
  int  i,j;
  double sum;

  for (i=1; i<=upper->units; i++) {
    sum = 0;
    for (j=0; j<=lower->units; j++) {
      sum += upper->weight[i][j] * lower->output[j];
    }
    upper->output[i] = 1 / (1 + exp(-net->sigmoidGain * sum));
  }
}


void propagateThroughNet(Net* net)
{
  int l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    propagateThroughLayer(net, net->layer[l], net->layer[l+1]);
  }
}


/***********************************
	Computing error and changing weights
	using backpropagation. 
***********************************/


void computeOutputError(Net* net, double* target)
{
  int  i;
 double out, err;
   
  net->totalError = 0;
  for (i=1; i<=net->outputLayer->units; i++) {
    out = net->outputLayer->output[i];
    err = target[i-1]-out;
    net->outputLayer->error[i] = net->sigmoidGain * out * (out) * err;
    net->totalError += 0.5 * sqr(err);
  }
}


void backpropagateThroughLayer(Net* net, Layer* upper, Layer* lower)
{
  int  i,j;
  double out, err;
   
  for (i=1; i<=lower->units; i++) {
    out = lower->output[i];
    err = 0;
    for (j=1; j<=upper->units; j++) {
      err += upper->weight[j][i] * upper->error[j];
    }
    lower->error[i] = net->sigmoidGain * out * (1-out) * err;
  }
}


void backpropagateThroughNet(Net* net)
{
  int l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    backpropagateThroughLayer(net, net->layer[l], net->layer[l-1]);
  }
}


void adjustWeights(Net* net)
{
  int  l,i,j;
  double out, err, dWeight;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=net->layer[l]->units; i++) {
      for (j=0; j<=net->layer[l-1]->units; j++) {
        out = net->layer[l-1]->output[j];
        err = net->layer[l]->error[i];
        dWeight = net->layer[l]->dWeight[i][j];
        net->layer[l]->weight[i][j] += net->learningRate * err * out + net->momentum * dWeight;
        net->layer[l]->dWeight[i][j] = net->learningRate * err * out;
      }
    }
  }
}


/***********************************
	Training  the network
***********************************/


void trainNet(Net* net)
{
  setInput(net, input);
  propagateThroughNet(net);
  getOutput(net, output);
 
 / **** FIX THIS ***/
  /**** get target ***/  
  computeOutputError(net, target);
  if (training) {
    backpropagateThroughNet(net);
    adjustWeights(net);
  }
}



/***********************************
	Main
***********************************/

int main()
{
  int epochNum = 0;
  Net  net;
  Bool stop;

  
  seedPseudoRandomGenerator();
  generateNetwork(&net);
  assignRandomWeights(&net);
  /* file for storing weights */
  outputFile = fopen("BP_Results.txt", "w");

  stop = FALSE;
  do {
    trainNet(&net);
    epochNum++;
    if (testError < errorCriteria) {
      fprintf(outputFile, "Saving weights");
      saveWeights(&net);
      stop = TRUE;
    } else if (epochNum > maxEpochs) {
       printf(outputFile, "Max epochs reached. Saving weights");
       saveWeights(&net);
       stop = TRUE;
    }
    
  } while (!stop);

  cleanUpTasks(&net);
  
  return 0;
}
