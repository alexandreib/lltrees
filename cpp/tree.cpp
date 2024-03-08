#include "tree.hpp"
#include "conf.hpp"

///////////////////////////////////////// Constructor / Destructor
template<class T>
tree<T>::~tree(){ 
    deleteTree(this->node_0);
    this->node_0 = NULL;
} 

template<class T> 
void tree<T>::deleteTree(node<T>* node){ 
    if (node == NULL) return; 
    this->deleteTree(node->l_node); 
    this->deleteTree(node->r_node); 
    delete node;
} 

///////////////////////////////////////// Fit Area
template<class T>
void tree<T>::fit(const data& tr, const std::vector<T>& target) {
    this->node_0 = new node<T>(tr.number_of_rows);
    std::vector<int> index(tr.number_of_rows);
    std::iota(index.begin(), index.end(), 0);
    this->_grow(*this->node_0, tr, target, index);
}

template<class T> 
void tree<T>::_grow(node<T>& pnode, const data& tr, const std::vector<T>& Y, const std::vector<int>& index) {
    //const data_type<T>& trs = static_cast <const data_type<T>&> (tr);
    if (pnode.level < conf_trees.max_depth) {
        for (int index_col = 0; index_col < tr.number_of_cols; index_col++) {
            // std::vector<double> unique_sorted = tr.get_column(index_col, index); //column;
            std::vector<double> unique_sorted;
            for(auto const &index_row : index) {
                // std::cout << "index_row:"<<index_row<< " " << Y[index_row] << std::endl;
                unique_sorted.push_back(tr.x[index_row * tr.number_of_cols + index_col]); 
            }
            std::sort(unique_sorted.begin(), unique_sorted.end());
            unique_sorted.erase(std::unique(unique_sorted.begin(), unique_sorted.end()), unique_sorted.end());

            if (unique_sorted.size() ==1 ) {break;}

            for(long unsigned int idx = 0; idx < unique_sorted.size() -1 ; idx++) {
                double threshold = (unique_sorted[idx] + unique_sorted[idx+1])/2;
                std::vector<T> l_Y, r_Y;
                for(auto const &index_row : index) {
                    if (tr.x[index_row * tr.number_of_cols + index_col] <= threshold) {
                        l_Y.push_back( Y[index_row]); 
                    }
                    else { 
                        r_Y.push_back( Y[index_row]); 
                    }
                }
                
                if (r_Y.size() < 1) {std::cout << "r_Y.size() < 1" << std::endl; break;}
                if (l_Y.size() < 1) {std::cout << "l_Y.size() < 1" << std::endl; break;}
                double l_crit = criterion_tree->get(l_Y);
                double r_crit = criterion_tree->get(r_Y);
                double impurity = (l_Y.size()/(double) index.size())* l_crit + (r_Y.size()/(double) index.size())*r_crit;
                //(!isnan( impurity )) && 
                // std::cout << impurity << std::endl;
                if (!isnan(impurity) && impurity < pnode.impurity && r_Y.size() > conf_trees.min_leaf_size && l_Y.size() > conf_trees.min_leaf_size  ) {
                    pnode.impurity = impurity;
                    pnode.threshold = threshold;
                    pnode.index_col = index_col;
                    pnode.isleaf = false;
                    // std::cout << l_crit  << ":" << impurity << ":"<< r_crit<< std::endl; 
                    // std::cout << l_Y.size()  << ":" << index.size() << ":"<< r_Y.size()<< std::endl;   
                // } else if (isnan( impurity)) {
                //     std::cout << "isnan"<< std::endl; 
                }                
            }  
        }  
    }
    if (pnode.isleaf == false) {
        std::vector<int> l_index, r_index;
        for(auto const &index_row : index) {
            if (tr.x[index_row * tr.number_of_cols +  pnode.index_col] <= pnode.threshold) {  
                l_index.push_back(index_row);
            }
            else {        
                r_index.push_back(index_row);
            }
        }         
        
        node<T>* l_node = new node<T>(pnode.level+1, ++this->id_node, l_index.size(), pnode.impurity);
        node<T>* r_node = new node<T>(pnode.level+1, ++this->id_node, r_index.size(), pnode.impurity);
        //  pnode.impurity= saved_impurity;
        // pnode.index_col = saved_col;
        // pnode.threshold = saved_threshold;
        // pnode.isleaf = false;  
        pnode.l_size = l_index.size();  
        pnode.r_size = r_index.size();
        pnode.set_children(l_node, r_node);
        
        // pnode.print();

        // std::cout<<"pnode.leaf_value:" << pnode.leaf_value<< "   index[0]:" <<  index[0]<< " index.size():" <<  index.size() << std::endl;
        // std::cout<<"pnode.leaf_value:" << pnode.leaf_value<<" pnode.index_col:" << pnode.index_col <<"pnode.threshold:" << pnode.threshold << std::endl;
        this->_grow(*l_node, tr, Y, l_index);
        this->_grow(*r_node, tr, Y, r_index);    
    } 
    // else 
    {
    
    pnode.leaf_value = this->get_leaf_value(Y, index);
    // pnode.print();
        // std::cout<<"pnode.leaf_value:" << pnode.leaf_value<< "   index[0]:" <<  index[0]<< " index.size():" <<  index.size() << std::endl;
        // std::cout<<"pnode.leaf_value:" << pnode.leaf_value<<" pnode.index_col:" << pnode.index_col <<" pnode.threshold:" << pnode.threshold << std::endl;
    }

}

template<> double tree<double>::get_leaf_value(const std::vector<double>& Y, const std::vector<int>& index) {
    double average = 0;
    for(auto const &index_row : index) {
        average = average + Y[index_row];
    }                
     return average / index.size(); 
}

template<> int tree<int>::get_leaf_value(const std::vector<int>& Y, const std::vector<int>& index) {
    std::unordered_map<int, int> freqMap; 
    for (long unsigned int i = 0; i < index.size(); i++) { freqMap[Y[i]]++;}  
    auto maxElement = max_element(freqMap.begin(), freqMap.end(), 
                    [](const auto& a, const auto& b) { 
                      return a.second < b.second; 
                  }); 
    return maxElement->first; 
}

///////////////////////////////////////// Predict Area
template<class T> 
std::vector<T> tree<T>::predict(const data &d) {  
    std::vector<T> pred;
    for (int index_row = 0; index_row < d.number_of_rows; index_row ++){
        pred.push_back(this->predict_row(d.x + index_row * d.number_of_cols));
    }
    return pred;
}

template<class T> 
T tree<T>::predict_row(const double* row) const {  
    return this->_traverse(*this->node_0, row);
}

template<class T> 
T tree<T>::_traverse(const node<T>& pnode, const double * row) const {
    if (pnode.isleaf){ 
        return pnode.leaf_value; 
    }
    if (*(row + pnode.index_col) <= pnode.threshold) {
        return this->_traverse(pnode.get_l_children(), row);
    } else {
        return this->_traverse(pnode.get_r_children(), row);
    }
}

///////////////////////////////////////// Print Area
template<class T> void tree<T>::print() {
    this->_print_tree(*this->node_0);
}

template<class T> void tree<T>::_print_tree(node<T>& node) {
    if (!node.isleaf) {
        node.print();
        this->_print_tree(node.get_l_children());
        this->_print_tree(node.get_r_children());
    }
    else {
        node.print();
    }
}

template<class T> void tree<T>::print_node_0() {
    this->node_0->print();
}

template class tree<int>;  // Explicit instantiation
template class tree<double>;  // Explicit instantiation
