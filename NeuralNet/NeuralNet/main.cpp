//
//  main.cpp
//  NeuralNet
//
//  Created by Christie Mathews on 10/5/15.
//  Copyright (c) 2015 Christie Mathews. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// * * * * * * * * * * * * class Neuron * * * * * * * * * * * * * * *

class Neuron {
public:
    //tell Neuron how many neurons there are in the next layer
    Neuron(int numOutputs, unsigned myIndex);
    void setOutputVal(double val){ m_outputVal = val; }
    double getOutputVal(void) const{return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double m_outputVal;
    static double randomWeight(void) { return rand()/ double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    //Connection contains weight & deltaWeight
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
    static double eta; //(0->1)
    static double alpha; //(0->n)
};


// * * * * * * * * * * * * Net Declaration * * * * * * * * * * * * * * *
double Neuron::eta = 0.15; //learning rate
double Neuron::alpha = 0.5; //momentum

Neuron::Neuron(int numOutputs, unsigned myIndex){
    //c -> connection
    for (unsigned c = 0; c < numOutputs; ++c){
        //you could make connection a class with a constructor that
        //gives Connection a random weight but we can do it here.
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    //loop through neurons in prev layer.
    //include bias neuron
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputVal = transferFunction(sum);
}

double Neuron::transferFunction(double x){
    //tanh - output range [-1.0 -> 1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    //tanh derivative
    return 1.0 - x * x;
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer){
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        //individual input, magnified by gradient and train rate
        double newDeltaWeight = eta //training rate 0-slow .2-medium 1-reckless
                              * neuron.getOutputVal() * m_gradient + alpha //alpha is momentum rate
                              * oldDeltaWeight;                            //0 is no/ .5 is moderate momentum
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

// * * * * * * * * * * * * class Net * * * * * * * * * * * * * * *

class Net{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};


// * * * * * * * * * * * * Net Declaration * * * * * * * * * * * * * * *

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        //typedef constructor Layer()
        m_layers.push_back(Layer());
        //if last layer, numoutputs = 0; else, numOutputs = topology[nextLayer]
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum+1];
        
        // <= because we want a bias neuron at each layer
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            //gives you the last element in the container
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "made a Neuron" << endl;
        }
        //make bias neuron 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals){
    //check that the # of neurons and # of inputs match
    assert(inputVals.size()==m_layers[0].size() - 1);
    //assign input values to input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //forward propagate:
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum-1];
        //for each nuron in a Layer
        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
            //class neuron's feedForward function
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals){
    //Calcualte overall net error (RMS of output errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error); //RMS
    
    //implement a recent average measurement; error indication of how well the net is doing/being trained.
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
    / (m_recentAverageSmoothingFactor + 1.0);
    
    
    //calc output layer gradients
    
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    //calculate gradients on hidden layers
    
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        
        for(unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    //for all layers from outputs to first hidden layer, update connection weights
    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        
        for(unsigned n = 0; n < layer.size() - 1; ++n){
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(vector<double> &resultVals) const{
    resultVals.clear();
    for(unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

int main(){
    //create Neural Net
    //eg { 3, 2, 1 }
    vector<unsigned> topology;
    topology.push_back(3); //3 input neurons
    topology.push_back(2); //2 hidden neurons
    topology.push_back(1); //1 output neuron
    Net myNet(topology);
    
//    Train
    vector<double> inputVals;
    myNet.feedForward(inputVals);
    
    vector<double> targetVals;
    myNet.backProp(targetVals);
    
    //Use Neural Net
    vector<double> resultVals;
    myNet.getResults(resultVals);
    
    
}
