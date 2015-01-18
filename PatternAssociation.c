/***********************************
	Pattern association network
	Trains a two-layer network to associate a pattern of n inputs 
	to a pattern of n outputs.

	17 Jan 2015	
	B. Laden
	
	Based on work by D.E. Rumelhart, J.L. McClelland
	"Explorations on Parallel Distributed Processing"
 
    WORK IN PROGRESS....

***********************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/***********************************
	Global Declarations
***********************************/

#define min(x,y)      ((x)<(y) ? (x) : (y))
#define max(x,y)      ((x)>(y) ? (x) : (y))

typedef int           Bool;

#define false         0
#define true          1


int maxEpochs 	= 10000; /* ensures program terminates if errorCriterion isn't met */
int epochNum 	= 0;   /* keeps track of the current epoch */

double totalSumSquares = 0.0;
double errorCriterion = 0.1;

double* activation;
double* target;
double* weight;
double* calculatedOutput;
double* error;
double* epsilon;
double* bias;
double* noise;

/* flags for activation function type */
Bool linearThreshold 	= false;
Bool sigmoid 			= true;
Bool linear 			= false;

/* allows turning off weight changes */
Bool learningFlag = true;

/***********************************
   Network specific code
   TBD: Read in a configuration and pattern files
***********************************/
int numInputs = 8;
int numUnits = 8;
int numPatterns = 3;

setUpNetwork {

}

/***********************************
   Random routines
***********************************/

void initializeRandomSeed()
{
  srand(4711);
}


int randomValueInteger(int low, int high)
{
  return rand() % (high-low + 1) + low;
}      


double randomValueDouble(double low, double high)
{
  return ((double) rand() / HUGE_VAL) * (high-low) + low;
}      

/***********************************
   logistic 
***********************************/

logistic(x) {
  return (1.0/(1.0 + exp( -x)));
}

/***********************************
   probability 
   Returns 1.0 with a probability equal to the value of its argument
   rnd returns a uniformly distributed number between 0 and 1
***********************************/

probability(x){
 if (rnd() < x)
   return(1);
 else
 	return(0);
}

/***********************************
	runTrial
	Runs only one trial
***********************************/

runTrial() {
    setupPatterns();
    computeOutput();
    computeError();
    summaryStatistics();  
}


/***********************************
	computeOutput
    Calculates the activation for each output unit
    Options include linear, linear threshold, sigmoid,  or stochastic
***********************************/

computeOutput() {
	for (i = numInputs; i < numUnits; i++) {
	
		/* figure net input to each output unit */
		calculatedOutput[i] = bias[i];
		for (j = 0; j < numInputs; j++){
			calculatedOutput[i] += activation[j]*weight[i][j];	
		}	
		/* Apply activation scheme */
		if (linear) {
			activation[i] = calculatedOutput[i];
		} else if (linearThreshold){  
		    if (calculatedOutput[i] > 0)
		    	activation[i] = 1.0;
		    else	
		    	activation[i] = 0.0;
		} else if (sigmoid) {  
			activation[i] = logistic(calculatedOutput[i]);
		} else { /*  stochastic mode */
			activation[i] = probabilty(logistic(calculatedOutput[i]);
		
		}		
	}
}


/***********************************
	computeError
    Calculates the error for a trial
***********************************/

computeError() {
	/* i for output units, t for target */
	for (i = numInputs, t = 0; i < numUnits; t++; i++;){
	     error[i] = target[t] - activation[i];
	}
}


/***********************************
	changeWeights
    This is the "learning" phase. It uses least mean squares, 
    which is also known as the delta rule.
***********************************/

changeWeights() {
	for (i = numInputs; i < numUnits; i++){
		for (j = 0; j < numInputs; j++){
			weight[i][j] += epsilon[i][j]*error[i]*activation[j];
	    }
	bias[i] += bepsilon[i]*error[i];	
	}
}


/***********************************
	setupPatterns
	Populate input and output pattern structures 
***********************************/

setupPatterns(){
	/* TBD */
}

/***********************************
	trainNetwork
	Sets pattern order, then runs trials for each pattern
	Patterns can be presented to the network sequentially or randomly permuted.
***********************************/

trainNetwork(char* c) {

	int* used;    /* keeps track of the already-used patterns */
	int* unused;  /* keeps track of the yet-to-be used patterns */
	int  npat;
    int  old;
    
    for (t = 0; t < maxEpochs; t++){
      	totalSumSquares = 0.0;
      	epochNum++;    
      	/* create a list of pattern numbers */
		for (i = 0; i < numPatterns; i++) {
          used[i] = i;
        }         
    	if (c == 'p'){ 
      		for (i = 0; i < numPatterns; i++){
         		npat = rnd() * (numPatterns - i) + i;
         		old = used[i];
         		used[i] = unused[npat];
         		used[npat] = old; 
     	 	}
   		} 
   		for (i = 0; i < npatternd; i++ {
        		patno = used[i];
        		runTrial();
        		if (learningFlag) changeWeights();  
  		}
    	if (totalSumSquares < errorCriterion) {
    		return;
    	}
   }
}

/***********************************
	summaryStatistics
	TBD
***********************************/

summaryStatistics(){
 	/* TBD */
 	puts("Summary statistics");
 	// report epochNum, totalSumSquares, errorCriterion
 }

int main(void){
	puts("App is in progress.");
	
	return 0;
}