#include <string>
#include <thread> 
#include <list>

namespace conf
{
    extern int number_of_threads;
    extern int verbose;
    extern std::string mode;
    extern std::string mode;
    extern std::list <int> idx_cat_cols;

    namespace tree 
    {
        extern int max_depth;
        extern long unsigned int min_leaf_size;
        extern std::string criterion_name;
    };
    
    namespace gbt
    {
        extern std::string metric_name;
        extern int epochs;
        extern double learning_rate;
    };
}