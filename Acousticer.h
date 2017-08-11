#ifndef ACOUSTICER_H
#define ACOUSTICER_H

//#define HIDDEN_SYMBOL  __attribute__((visibility("hidden")))

#include <string>
#include <vector>

//class HIDDEN_SYMBOL Acousticer{
class Acousticer{
    public:
        Acousticer() {};  

        Acousticer(const std::string aModelPath);
        ~Acousticer();
        
        static Acousticer *createInstance(const std::string aModelPath);

        bool get_load_graph_status();
        bool get_session_create_status();
        
        void destory();
        
        bool process(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &ouput, std::vector<std::vector<float> > &state_holder, std::vector<std::vector<float> > &cell_holder);

    private:
        std::string m_aModelPath;

        bool _load_graph_status;
        bool _session_create_status;

        //tensorflow::GraphDef _graph_def;
        //std::unique_ptr<tensorflow::Session> _session;
        void* _graph_def;
        void* _session;
};

#endif /* ACOUSTICER_H */
