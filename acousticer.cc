#include "Acousticer.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <unistd.h>

//#include "tensorflow/cc/ops/const_op.h"
//#include "tensorflow/cc/ops/image_ops.h"
//#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
//#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
//using tensorflow::string;
using tensorflow::int32;
using namespace std;

#define BATCH 16

Acousticer::Acousticer(const std::string aModelPath) {
    //class member initialization
    _session = NULL; 
    _graph_def = NULL;
    _load_graph_status = false; 
    _session_create_status = false;
    //start initialize tf
    const std::string m_aModelPath = aModelPath;
    _graph_def = (void*)new tensorflow::GraphDef;
    tensorflow::GraphDef* pgraph_def = (tensorflow::GraphDef*)_graph_def;
    Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), m_aModelPath, pgraph_def);
    _load_graph_status = load_graph_status.ok();
    _session = (void*)new std::unique_ptr<tensorflow::Session>;
    std::unique_ptr<tensorflow::Session>* p_session = (std::unique_ptr<tensorflow::Session>*)_session;
    p_session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*p_session)->Create(*pgraph_def);
    _session_create_status = session_create_status.ok();
}

Acousticer *Acousticer::createInstance(const std::string aModelPath) {
    Acousticer *acousticer = new Acousticer(aModelPath);
    if (!(acousticer->get_load_graph_status())) {
        delete acousticer;
        return NULL;
    }
    if (!(acousticer->get_session_create_status())) {
        delete acousticer;
        return NULL;
    }
    return acousticer;
}

bool Acousticer::get_load_graph_status() {
    return _load_graph_status;
}

bool Acousticer::get_session_create_status() {
    return _session_create_status;
}

void Acousticer::destory() {
    std::unique_ptr<tensorflow::Session>* p_session = (std::unique_ptr<tensorflow::Session>*)_session;
    (*p_session)->Close();
}

Acousticer::~Acousticer() {
    if ((void*)_session != NULL) {
        std::unique_ptr<tensorflow::Session>* p_session = (std::unique_ptr<tensorflow::Session>*)_session;
        delete p_session;
    }
    if ((void*)_graph_def != NULL) {
        tensorflow::GraphDef* p_graph_def = (tensorflow::GraphDef*)_graph_def;
        delete p_graph_def;
    }
}

bool Acousticer::process(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &ouput, std::vector<std::vector<float> > &state_holder, std::vector<std::vector<float> > &cell_holder) {

    std::unique_ptr<tensorflow::Session>* p_session = (std::unique_ptr<tensorflow::Session>*)_session;

    unsigned int phn_num = input.size();
    if (phn_num == 0){
        std::cout << "  input vector size is zero" << std::endl;
        return false;
    }

    unsigned int lab_dim = input[0].size();
    if (lab_dim != 921) {
        std::cout << "  we expect input lab vector dim is 921" << std::endl;
        std::cout << "  but the real dimension is " << lab_dim << std::endl;
        return false;
    }

    if (state_holder.size() != 3) {
        std::cout << "  state_holder dim is "<< state_holder.size() << " (expecet 3)" << std::endl;
        return false;
    }
    if (cell_holder.size() != 3) {
        std::cout << "  cell_holder dim is "<< cell_holder.size() << " (expecet 3)" << std::endl;
        return false;
    }

    for (unsigned int i=0; i<3; i++) {
        if (state_holder[i].size() != 256) {
            std::cout << "  state_holder dim for row " << i+1 << " is "<< state_holder[i].size() << " (expecet 256)" << std::endl;
            return false;
        }
        if (cell_holder[i].size() != 512) {
            std::cout << "  cell_holder dim for row " << i+1 << " is "<< cell_holder[i].size() << " (expecet 512)" << std::endl;
            return false;
        }
    }

    //initialize zero state and cell tensor
    tensorflow::Tensor state_acc_0_placeholder_tensor(tensorflow::DT_FLOAT, {1, 256});
    tensorflow::Tensor state_acc_1_placeholder_tensor(tensorflow::DT_FLOAT, {1, 256});
    tensorflow::Tensor state_acc_2_placeholder_tensor(tensorflow::DT_FLOAT, {1, 256});
    tensorflow::Tensor cell_acc_0_placeholder_tensor(tensorflow::DT_FLOAT, {1, 512});
    tensorflow::Tensor cell_acc_1_placeholder_tensor(tensorflow::DT_FLOAT, {1, 512});
    tensorflow::Tensor cell_acc_2_placeholder_tensor(tensorflow::DT_FLOAT, {1, 512});

    auto state_acc_0_tensor = state_acc_0_placeholder_tensor.tensor<float, 2>();
    auto state_acc_1_tensor = state_acc_1_placeholder_tensor.tensor<float, 2>();
    auto state_acc_2_tensor = state_acc_2_placeholder_tensor.tensor<float, 2>();
    auto cell_acc_0_tensor = cell_acc_0_placeholder_tensor.tensor<float, 2>();
    auto cell_acc_1_tensor = cell_acc_1_placeholder_tensor.tensor<float, 2>();
    auto cell_acc_2_tensor = cell_acc_2_placeholder_tensor.tensor<float, 2>();

    for(int i = 0; i < 1; ++i){
        for(int j = 0; j < 256; ++j){
            state_acc_0_tensor(i, j) = state_holder[0][j];
            state_acc_1_tensor(i, j) = state_holder[1][j];
            state_acc_2_tensor(i, j) = state_holder[2][j];
        }
        for(int k = 0; k < 512; ++k){
            cell_acc_0_tensor(i, k) = cell_holder[0][k];
            cell_acc_1_tensor(i, k) = cell_holder[1][k];
            cell_acc_2_tensor(i, k) = cell_holder[2][k];
        }
    }

    //initialize input and hist tensor
    //string norm_label_input_layer = "create_inputs/padding_fifo_queue_DequeueMany:0";
    //string start_mask_input_layer = "create_inputs/padding_fifo_queue_DequeueMany:3";
    const string norm_label_input_layer = "create_model/Model/Placeholder_6";
    const string start_mask_input_layer = "create_model/Model/Placeholder_7";

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, {1, BATCH, 921});
    tensorflow::Tensor start_mask_tensor(tensorflow::DT_FLOAT, {1, 1});
    auto mask_tensor = start_mask_tensor.tensor<float, 2>();
    mask_tensor(0, 0) = 0;
  
    //state and cell, no input, only output
    const string state_acc_0_pholder_input_layer = "create_model/Model/Placeholder";   //"create_model/Model/stateholder_0";
    const string state_acc_1_pholder_input_layer = "create_model/Model/Placeholder_2";    //"create_model/Model/stateholder_1";
    const string state_acc_2_pholder_input_layer = "create_model/Model/Placeholder_4";    //"create_model/Model/stateholder_2";
    const string cell_acc_0_pholder_input_layer = "create_model/Model/Placeholder_1";   //"create_model/Model/cellholder_0";
    const string cell_acc_1_pholder_input_layer = "create_model/Model/Placeholder_3";     //"create_model/Model/cellholder_1";
    const string cell_acc_2_pholder_input_layer = "create_model/Model/Placeholder_5";     //"create_model/Model/cellholder_2";

    const string output_layer = "rnn/feedforward_3/Reshape_1";
    //state and cell, output from a LSTM black box, assign to "state_acc_0_pholder_input_layer" etc.
    /* // for batch = 1
    string state_acc_0_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell/Cell0/MyLSTMCell/MatMul_1";
    string state_acc_1_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell/Cell1/MyLSTMCell/MatMul_1";
    string state_acc_2_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell/Cell2/MyLSTMCell/MatMul_1";
    string cell_acc_0_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell/Cell0/MyLSTMCell/add_3";
    string cell_acc_1_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell/Cell1/MyLSTMCell/add_3";
    string cell_acc_2_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell/Cell2/MyLSTMCell/add_3";
    */
    // for batch = 16
    const string state_acc_0_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell_15/Cell0/MyLSTMCell/MatMul_1";
    const string state_acc_1_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell_15/Cell1/MyLSTMCell/MatMul_1";
    const string state_acc_2_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell_15/Cell2/MyLSTMCell/MatMul_1";
    const string cell_acc_0_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell_15/Cell0/MyLSTMCell/add_3";
    const string cell_acc_1_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell_15/Cell1/MyLSTMCell/add_3";
    const string cell_acc_2_pholder_output_layer = "rnn/lstmp_stack3_2/MultiRNNCell_15/Cell2/MyLSTMCell/add_3";

    auto mytensor = input_tensor.tensor<float, 3>();

    unsigned int batch_seg_num = phn_num / BATCH;
    //unsigned int remain_num = phn_num % 16;

    unsigned int batch_counter = 0;
    
    
    ///ouput.reserve(1<<24); //16M allocation
    ouput.clear();

    std::vector<tensorflow::Tensor> sess_outputs;
    /// sess_outputs.reserve(1<<20); //16M allocation

    //feed 16sample per run
    for (unsigned int ii=0; ii<=batch_seg_num; ii++) {
        
        if (batch_counter*BATCH >= phn_num) break;

        for (int j = 0; j < BATCH; j++)
            for (int k = 0; k < 921; ++k) {
                if (batch_counter*BATCH + j < phn_num) {
                    mytensor(0, j, k) = input[batch_counter*BATCH + j][k];
                } else {
                    mytensor(0, j, k) = 0.0;
                }
            }

        /// std::vector<tensorflow::Tensor> sess_outputs;
        sess_outputs.clear();
        //for (unsigned int i=0; i<tmp_phn_bound; i++) {
                    ///std::cout << "leodebug  inner AAAAcoustic class, start run() ===" << std::endl;
        Status run_status = (*p_session)->Run({{norm_label_input_layer, input_tensor},\
                                        {start_mask_input_layer, start_mask_tensor},\
                                        {state_acc_0_pholder_input_layer, state_acc_0_placeholder_tensor},\
                                        {state_acc_1_pholder_input_layer, state_acc_1_placeholder_tensor},\
                                        {state_acc_2_pholder_input_layer, state_acc_2_placeholder_tensor},\
                                        {cell_acc_0_pholder_input_layer, cell_acc_0_placeholder_tensor},\
                                        {cell_acc_1_pholder_input_layer, cell_acc_1_placeholder_tensor},\
                                        {cell_acc_2_pholder_input_layer, cell_acc_2_placeholder_tensor}},
                                    {output_layer, \
                                    state_acc_0_pholder_output_layer, \
                                    state_acc_1_pholder_output_layer, \
                                    state_acc_2_pholder_output_layer, \
                                    cell_acc_0_pholder_output_layer, \
                                    cell_acc_1_pholder_output_layer, \
                                    cell_acc_2_pholder_output_layer},
                                    {}, &sess_outputs);
                ///std::cout << "leodebug  inner AAAAAAcoustic class, end ***" << std::endl;
        if (!run_status.ok()) {
            std::cout << "Running model failed: " << std::endl;
            return false;
        }
        //usleep(200);
        state_acc_0_placeholder_tensor = sess_outputs[1];
        state_acc_1_placeholder_tensor = sess_outputs[2];
        state_acc_2_placeholder_tensor = sess_outputs[3];
        cell_acc_0_placeholder_tensor = sess_outputs[4];
        cell_acc_1_placeholder_tensor = sess_outputs[5];
        cell_acc_2_placeholder_tensor = sess_outputs[6];

        /*
        auto state_acc_0_out = state_acc_0_placeholder_tensor.tensor<float, 2>();
        auto state_acc_1_out = state_acc_1_placeholder_tensor.tensor<float, 2>();
        auto state_acc_2_out = state_acc_2_placeholder_tensor.tensor<float, 2>();
        auto cell_acc_0_out = cell_acc_0_placeholder_tensor.tensor<float, 2>();
        auto cell_acc_1_out = cell_acc_1_placeholder_tensor.tensor<float, 2>();
        auto cell_acc_2_out = cell_acc_2_placeholder_tensor.tensor<float, 2>();


        for(int j = 0; j < 256; ++j){
            state_holder[0][j] = state_acc_0_out(0, j);
            state_holder[1][j] = state_acc_1_out(0, j);
            state_holder[2][j] = state_acc_2_out(0, j);
        }
        for(int k = 0; k < 512; ++k){
            cell_holder[0][k] = cell_acc_0_out(0, k);
            cell_holder[1][k] = cell_acc_1_out(0, k);
            cell_holder[2][k] = cell_acc_2_out(0, k);
        }
        */

        tensorflow::Tensor& output_elements = sess_outputs[0];

        auto output_elements_tensor = output_elements.tensor<float, 3>();
        for(unsigned int i=0; i<BATCH; i++) {
            if (batch_counter*BATCH + i < phn_num) {
                std::vector<float> single_vec;
                for(unsigned int j=0; j<50; j++) 
                    single_vec.push_back(output_elements_tensor(0, i, j));
                ouput.push_back(single_vec);
            } else {
                break;
            }
        }

        ++batch_counter;
    }

    auto state_acc_0_out = state_acc_0_placeholder_tensor.tensor<float, 2>();
    auto state_acc_1_out = state_acc_1_placeholder_tensor.tensor<float, 2>();
    auto state_acc_2_out = state_acc_2_placeholder_tensor.tensor<float, 2>();
    auto cell_acc_0_out = cell_acc_0_placeholder_tensor.tensor<float, 2>();
    auto cell_acc_1_out = cell_acc_1_placeholder_tensor.tensor<float, 2>();
    auto cell_acc_2_out = cell_acc_2_placeholder_tensor.tensor<float, 2>();

    for(int j = 0; j < 256; ++j){
        state_holder[0][j] = state_acc_0_out(0, j);
        state_holder[1][j] = state_acc_1_out(0, j);
        state_holder[2][j] = state_acc_2_out(0, j);
    }
    for(int k = 0; k < 512; ++k){
        cell_holder[0][k] = cell_acc_0_out(0, k);
        cell_holder[1][k] = cell_acc_1_out(0, k);
        cell_holder[2][k] = cell_acc_2_out(0, k);
    }

    return true;
}
