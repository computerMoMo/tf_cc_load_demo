//
// Created by jeffly on 17-8-10.
//
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


using namespace std;
using namespace tensorflow;

int main()
{
    const string pathToGraph = "demo_model/demo.meta";
    const string checkpointPath = "demo_model/demo";
    auto session = NewSession(SessionOptions());
    if (session == nullptr) {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

// Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

// Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

// Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    std::string order;
    cout<<"continue input..."<<endl;
    cin >> order;
//    input
    std::vector<std::pair<string, Tensor>> input;
    tensorflow::TensorShape inputshape;
    inputshape.InsertDim(0,1);
    Tensor a(tensorflow::DT_INT32,inputshape);
    Tensor b(tensorflow::DT_INT32,inputshape);

    auto a_map = a.tensor<int,1>();
    a_map(0) = 2;
    auto b_map = b.tensor<int,1>();
    b_map(0) = 3;
    input.emplace_back(std::string("a"), a);
    input.emplace_back(std::string("b"), b);

    cout<<"continue run..."<<endl;
    cin >> order;

    std::vector<tensorflow::Tensor> answer;
    status = session->Run(input, {"res"}, {}, &answer);
    cout<<"run over"<<endl;
    cin>>order;
    Tensor result = answer[0];
    auto result_map = result.tensor<int,1>();
    cout<<"result: "<<result_map(0)<<endl;
    return 0;
}
