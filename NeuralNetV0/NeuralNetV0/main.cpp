//
//  main.cpp
//  NeuralNet
//
//  Created by Christie Mathews on 10/5/15.
//  Copyright (c) 2015 Christie Mathews. All rights reserved.
//

#include <iostream>
#include <vector>
using namespace std;


class Neuron {};

typedef vector<Neuron> Layer;

class Net{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
};

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        //typedef constructor Layer()
        m_layers.push_back(Layer());
        
        // <= because we want a bias neuron at each layer
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            //gives you the last element in the container
            m_layers.back().push_back(Neuron());
            cout << "made a Neuron" << endl;
        }
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
    
    //Train
    vector<double> inputVals;
    myNet.feedForward(inputVals);
    
    //    vector<double> targetVals;
    //    myNet.backProp(targetVals);
    //
    //    //Use Neural Net
    //    vector<double> resultVals;
    //    myNet.getResults(resultVals);
    
    
}
