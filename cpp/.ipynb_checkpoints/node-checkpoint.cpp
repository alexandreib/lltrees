#include <map>
#include "node.hpp"
#include "conf.hpp"

template <class T> 
void node<T>::set_children(node<T>* left, node<T>* right) 
{
    this->l_node = left;
    this->r_node = right;
}

template <class T>
node<T>& node<T>::get_l_children() const 
{
    return *l_node;
}

template <class T> 
node<T>& node<T>::get_r_children() const 
{
    return *r_node;
}

template <class T>
void node<T>::print() 
{
    std::cout << "**********" << std::endl;
    std::cout << "Node id : " << this->id_node << std::endl;
    std::cout << "Node level : " << level << std::endl;
    std::cout << "Node impurity : " << impurity << std::endl;
    std::cout << "Node index_col : " << index_col << std::endl;
    std::cout << "Node threshold : " << threshold << std::endl;
    std::cout << "Node isleaf : " << std::boolalpha << isleaf << std::endl;
    std::cout << "Node leaf_value : " << leaf_value << std::endl;
    std::cout << "Node size : " << size << std::endl;  
    for (const auto& pair : this->probas) 
    { 
        std::cout<< pair.first <<" : " << pair.second << std::endl;  
    }  
    std::cout << "**********" << std::endl;
}

template<class T> 
template<class U> 
U node<T>::get_leaf_value() const
{
    __builtin_unreachable();
}

template<> 
template<> 
double node<double>::get_leaf_value() const
{
    return this->leaf_value;
}

template<> 
template<> 
int node<int>::get_leaf_value() const
{
    return this->leaf_value;
}

template<> 
template<> 
std::map<int, double> node<int>::get_leaf_value() const
{
    return this->probas;
}

template int node<int>::get_leaf_value() const;  
template double node<double>::get_leaf_value() const; 

template int node<double>::get_leaf_value() const;  
template double node<int>::get_leaf_value() const; 

template std::unordered_map<int, double> node<int>::get_leaf_value() const; 

template<> 
void node<double>::set_node_value(const XY & tr, const std::vector<double>& Y, const std::vector<int>& index) 
{
    double sum = 0;
    for(auto const &index_row : index)
    {
        sum = sum + Y[index_row];
    }                

    double div = index.size();
    if (conf::mode == "classic_classification") 
    {
        const std::vector<double> & proba =  tr.get_vector_proba();
        div = 0;
        for(auto const &index_row : index)
        {
            div = div + proba[index_row] * ( 1 - proba[index_row] );
        }   
    }
    this->leaf_value = sum / div; 
    // std::cout << "this->leaf_value : "  << this->leaf_value << std::endl;
}

template<> 
void node<double>::set_probas(const std::vector<double>& Y, const std::vector<int>& index) 
{
    __builtin_unreachable();
}

template<> 
void node<int>::set_probas(const std::vector<int>& Y, const std::vector<int>& index) 
{
    for (long unsigned int idx : index) 
    { 
        if (this->probas.find(Y[idx]) == this->probas.end())
        {
            this->probas[Y[idx]] = 1 ;
        }
        else
        {
            this->probas[Y[idx]] += 1 ;
        }
    }
    for (auto const &prob : this->probas) 
    {
        this->probas[prob.first] = (double) prob.second /  (double) index.size();
        // std::cout << prob.first << ":"<< this->probas[prob.first] << std::endl;
    }
}

template<> 
void node<int>::set_node_value(const XY & tr, const std::vector<int>& Y, const std::vector<int>& index) 
{
    this->set_probas(Y, index);
    double max_proba = 0;
    for (const auto & pair : this->probas) 
    { 
        if (pair.second > max_proba) 
        { 
            max_proba = pair.second;
            this->leaf_value = pair.first; 
        } 
    }  
}

template class node<int>;  // Explicit instantiation
template class node<double>;  // Explicit instantiation