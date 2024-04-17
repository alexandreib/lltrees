#include <list>
#include <map>
#include "conf.hpp"
#include "gbt.hpp"
#include "factories.hpp"

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// base_gbt //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
inline void base_gbt::print_epoch_log(int & epoch, double & metric_tr, double & metric_va, double residuals) 
{
    std::cout << "Epoch : " << std::setw(5) << epoch << " Metric Train : " << std::setw(7) << metric_tr << " Metric va : " << std::setw(7) << metric_va << " Residuals (sum) : " << std::setw(7) << residuals << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Gbt template ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
template<class T>
gbt<T>::gbt() 
{
}

template<class T>
gbt<T>::~gbt() 
{
    for (auto p : this->trees) 
    {
        delete p;
    } 
    this->trees.clear();
}

template<class T>
void gbt<T>::print() 
{
    for (long unsigned int i =0; i < this->trees.size(); i++)
    {
        std::cout << "Tree : " << i << std::endl;
        this->trees[i]->printBT();
    }
}

///////////////////////////////// Save / Load ////////////////////////////////
template<class T>
void gbt<T>::save() 
{    
    std::ofstream myfile("trees.txt");
    myfile << conf::verbose << ":"  
        << conf::mode << ":"  
        << conf::tree::criterion_name << ":"  
        << conf::gbt::metric_name << ":" 
        << conf::gbt::epochs << ":"  
        << conf::gbt::learning_rate << ":" 
        << conf::number_of_threads  << "\n";
    myfile << conf::tree::max_depth << ":"  << conf::tree::min_leaf_size << "\n";
    for (long unsigned int i =0; i < this->trees.size(); i++)
    {
        this->trees[i]->save(myfile);
        myfile << "\n";
    }
    myfile.close();
}

template<class T>
void gbt<T>::load() 
{    
    std::ifstream myfile("trees.txt");
    std::string delimiter = ":";
    std::string line;
    std::getline(myfile, line);
    std::string token = line.substr(0, line.find(delimiter));
    line.erase(0, token.size() + delimiter.size());
    conf::verbose = std::stoi(token);

    conf::mode = line.substr(0, line.find(delimiter));
    line.erase(0, conf::mode.size() + delimiter.size());
    
    conf::tree::criterion_name = line.substr(0, line.find(delimiter));
    line.erase(0, conf::tree::criterion_name.size() + delimiter.size());
    
    conf::gbt::metric_name = line.substr(0, line.find(delimiter));
    line.erase(0, conf::gbt::metric_name.size() + delimiter.size());
    
    token = line.substr(0, line.find(delimiter));
    line.erase(0, token.size() + delimiter.size());
    conf::gbt::epochs = std::stoi(token);

    token = line.substr(0, line.find(delimiter));
    line.erase(0, token.size() + delimiter.size());
    conf::gbt::learning_rate = std::stod(token);
    
    token = line.substr(0, line.find(delimiter));
    conf::number_of_threads = std::stoi(token);

    std::getline(myfile, line);
    token = line.substr(0, line.find(delimiter));
    line.erase(0, token.size() + delimiter.size());
    conf::tree::max_depth = std::stoi(token);

    token = line.substr(0, line.find(delimiter));
    conf::tree::min_leaf_size = std::stoi(token);
    
    while (std::getline(myfile, line))
    {             
        tree<T>* my_tree = new tree<T>();
        my_tree->load(line);
        this->trees.push_back(my_tree);
    }
    myfile.close();
}


template class gbt<int>;  // Explicit instantiation
template class gbt<double>;  // Explicit instantiation

//////////////////////////////////////////////////////////////////////////////
////////////////////////// abstract_classification ///////////////////////////
//////////////////////////////////////////////////////////////////////////////
inline std::vector<double> abstract_classification::softmax(std::vector<double> odds) const
{   
    // std::cout <<__PRETTY_FUNCTION__ << std::endl;
    std::vector<double> probas(odds.size());
    int number_of_rows = odds.size() / this->classes.size();
    for (int row_idx = 0; row_idx < number_of_rows;  row_idx ++) 
    {   
        double sum=0;
        for(long unsigned int idx_cla = 0;  idx_cla < this->classes.size(); idx_cla++) 
        {                
            sum += std::exp( odds[idx_cla + row_idx * this->classes.size()]); 
        }
        for(long unsigned int idx_cla = 0;  idx_cla < this->classes.size(); idx_cla++) 
        {    
            double log_odd_pred =  std::exp(odds[idx_cla + row_idx * this->classes.size()]); 
            probas[idx_cla + row_idx * this->classes.size()] = log_odd_pred / sum;
        }
    }
    return probas;
}

inline std::vector<int> abstract_classification::extract_pred_from_proba(const std::vector<double> probas) const
{
    int row_numbers = probas.size() / this->classes.size();
    std::vector<int> preds(row_numbers);
    for (int row_idx = 0; row_idx < row_numbers;  row_idx ++) 
    {
        std::list<double> l;
        for(long unsigned int idx_classe = 0; idx_classe <  this->classes.size(); idx_classe++)
        {    
            l.push_back(probas[idx_classe + row_idx * this->classes.size()]);
        }

        preds[row_idx] = *std::next(this->classes.begin(), std::distance(l.begin(), std::max_element(l.begin(), l.end())));
    }
    return preds;
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////// abstract_classic_boost ///////////////////////////
//////////////////////////////////////////////////////////////////////////////
std::vector<double> abstract_classic_boost::get_residuals() const
{   
    return residuals_saved;
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////// adaboost_classification ///////////////////////////
//////////////////////////////////////////////////////////////////////////////
void adaboost_classification::fit(XY & tr, const XY & va) 
{    
    ThreadPool * pool = new ThreadPool(conf::number_of_threads);
    std::cout<< "Gbt_classification fit" << std::endl;  
    const int* y_tr = tr.get_y<int>();
    const int* y_va = va.get_y<int>();
    
    this->classes.insert(y_tr, y_tr + tr.number_of_rows); 
    std::cout<<"All the distinct element for classification in sorted order are: ";
    for(auto it:this->classes) std::cout<<it<<" "; std::cout << std::endl;

    std::unique_ptr<base_factory> factory = base_factory::get_instance();  
    std::shared_ptr<base_criterion> criterion = factory->Criterion(); 
    std::shared_ptr<base_metrics> metric = factory->Metric(); 
    
    std::vector<double> weights(tr.number_of_rows, 1.0 / tr.number_of_rows);
        
    std::vector<int> vec_y_tr;
    vec_y_tr.insert(vec_y_tr.end(), y_tr, y_tr + tr.number_of_rows); 

    // https://www.intel.com/content/www/us/en/docs/onedal/developer-guide-reference/2024-1/adaboost-multiclass.html
    // https://www.intlpress.com/site/pub/files/_fulltext/journals/sii/2009/0002/0003/SII-2009-0002-0003-a008.pdf
    for (int epoch = 1; epoch < conf::gbt::epochs + 1; epoch++){        
        tree<int>* my_tree = new tree<int>(criterion, pool);
        my_tree->fit(tr, vec_y_tr, weights);
        
        std::vector<int> pred_tr = my_tree->predict<int>(tr);

        double err = 0;
        std::vector<int> loss (tr.number_of_rows, 0);

        double sum_weight = std::accumulate(weights.begin(), weights.end(), 0.0);

        for (int idx =0; idx < tr.number_of_rows; idx ++)
        {
            if (pred_tr[idx] != y_tr[idx])
            {
                loss[idx] = 1;
                err += weights[idx] / sum_weight;
            }
            else 
            {
                loss[idx] = -1;                
            }
        }
        double alpha = std::log((1-err) / err) * 0.5;//std::log(this->classes.size()-1);
        this->trees.push_back(my_tree);
        this->models_weights.push_back(alpha);
        
        for (int idx =0; idx < tr.number_of_rows; idx ++)
        { 
            weights[idx] *= std::exp(alpha * loss[idx]);                
        }
        sum_weight = std::accumulate(weights.begin(), weights.end(), 0.0);       
        if (conf::verbose == 1) 
        {
            pred_tr = this->get_predict(tr);
            std::vector<int> pred_va = this->get_predict(va);
            
            double metric_tr = metric->get(pred_tr, y_tr);
            double metric_va = metric->get(pred_va, y_va);
            
            this->print_epoch_log(epoch, metric_tr, metric_va );
        }   
    }    
    delete pool;
}

std::vector<double> adaboost_classification::get_proba(const XY & d) const
{
    std::vector<double> preds;
    for (int index_row = 0; index_row < d.number_of_rows; index_row ++)
    {
        std::vector<double> row_pred (this->classes.size());
        for (long unsigned int model_idx = 0; model_idx < this->models_weights.size(); model_idx ++)
        {
            double model_weight = models_weights[model_idx];     
            tree<int>* my_tree =  this->trees[model_idx];
            for(auto classe : this->classes)
            {
                auto idx_classe = std::distance(this->classes.begin(), this->classes.find(classe));
                int tree_pred = my_tree->predict_row<int>(d.x + index_row * d.number_of_cols);
                
                if (tree_pred == classe ) 
                {
                    row_pred[idx_classe] += model_weight;
                }
            }
        }
        for (auto proba : row_pred) {preds.push_back(proba);}        
    }
    return this->softmax(preds);
}

boost::python::numpy::ndarray adaboost_classification::predict_proba(const XY & d) 
{
    std::vector<double> probas = this->get_proba(d);
    const double * data_ptr = probas.data();
    boost::python::tuple shape = boost::python::make_tuple(d.number_of_rows, this->classes.size());
    boost::python::tuple stride = boost::python::make_tuple(sizeof(double) * this->classes.size(), sizeof(double) );
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<double>();
    return boost::python::numpy::from_data(data_ptr, dt, shape, stride, boost::python::object()).copy();
}

std::vector<int> adaboost_classification::get_predict(const XY & d) const
{
    std::vector<double> probas = this->get_proba(d);
    return this->extract_pred_from_proba(probas);
}

boost::python::numpy::ndarray adaboost_classification::predict(const XY & d) 
{
    std::vector<int> preds = this->get_predict(d);
    return boost::python::numpy::from_data(preds.data(),
                                                boost::python::numpy::dtype::get_builtin<int>(), 
                                                boost::python::make_tuple(preds.size()), 
                                                boost::python::make_tuple(sizeof(int)), 
                                                boost::python::object()).copy();
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////// classic_classification ///////////////////////////
//////////////////////////////////////////////////////////////////////////////
void classic_classification::fit(XY & tr, const XY & va) 
{    
    ThreadPool * pool = new ThreadPool(conf::number_of_threads);
    std::cout<< "Gbt_classic_classification fit" << std::endl;  
    const int* y_tr = tr.get_y<int>();
    const int* y_va = va.get_y<int>();

    this->classes.insert(y_tr, y_tr + tr.number_of_rows); 
    std::cout<<"All the distinct element for classification in sorted order are: ";
    for(auto classe:this->classes) std::cout<<classe<<" "; std::cout << std::endl;

    std::unique_ptr<base_factory> factory = base_factory::get_instance();  
    std::shared_ptr<base_criterion> criterion = factory->Criterion(); 
    std::shared_ptr<base_metrics> metric = factory->Metric(); 

    std::vector<int> y_tr_one_hot_encoded(this->classes.size() * tr.number_of_rows, 0);
    for (int row_idx = 0; row_idx < tr.number_of_rows; row_idx ++ )
    {
        y_tr_one_hot_encoded[std::distance(this->classes.begin(), this->classes.find(y_tr[row_idx])) +  row_idx * this->classes.size()] = 1;
    }    
    
    std::vector<double> tr_odds(tr.number_of_rows * this->classes.size());
    std::vector<double> va_odds(va.number_of_rows * this->classes.size());
    for(auto classe : this->classes)
    {    
        double count = (double) std::count(y_tr, y_tr +  tr.number_of_rows, classe) ;
        double log_odd = std::log(count /  (double) (tr.number_of_rows - count));
        auto classe_idx = std::distance(this->classes.begin(), this->classes.find(classe));
        for (int idx_row = 0; idx_row < tr.number_of_rows; idx_row ++) 
        {     
            tr_odds[idx_row * this->classes.size() + classe_idx] = log_odd;
        }
        for (int idx_row = 0; idx_row < va.number_of_rows; idx_row ++) 
        {     
            va_odds[idx_row * this->classes.size() + classe_idx] = log_odd;
        }
        log_odds_classes.push_back(log_odd);
    }   
    
    std::vector<double> tr_probas = this->softmax(tr_odds);
    std::vector<double> va_probas = this->softmax(va_odds);

    std::vector<double> residuals(tr.number_of_rows);
    std::vector<double> probas_classe(tr.number_of_rows);
    for (int epoch = 1; epoch < conf::gbt::epochs + 1; ++ epoch)
    {   
        double sum_residuals =0;
        for(long unsigned int idx_classe = 0; idx_classe < this->classes.size(); idx_classe++)
        {    
            tree<double>* my_tree = new tree<double>(criterion, pool);
            for (int row_idx = 0; row_idx < tr.number_of_rows; row_idx ++)
            {
                probas_classe[row_idx] = tr_probas[row_idx * this->classes.size() + idx_classe];
                residuals[row_idx] = y_tr_one_hot_encoded[row_idx * this->classes.size() + idx_classe] - probas_classe[row_idx];
                sum_residuals += std::abs(residuals[row_idx]);
            }      
            tr.set_pred<double>(probas_classe);
                        
            my_tree->fit(tr, residuals);
            this->trees.push_back(my_tree);
            
            for (int row_idx = 0; row_idx < tr.number_of_rows; row_idx ++)
            {
                tr_odds[row_idx * this->classes.size() + idx_classe] +=  conf::gbt::learning_rate * my_tree->predict_row<double>(tr.x + row_idx * tr.number_of_cols);     
            }
            for (int row_idx = 0; row_idx < va.number_of_rows; row_idx ++)
            {
                va_odds[row_idx * this->classes.size() + idx_classe] +=  conf::gbt::learning_rate * my_tree->predict_row<double>(va.x + row_idx * va.number_of_cols);     
            }    
            
        }    
        this->residuals_saved.push_back(sum_residuals); 
        
        tr_probas = this->softmax(tr_odds);
        va_probas = this->softmax(va_odds);
        
        if (conf::verbose == 1) 
        {
            std::vector<int> pred_tr = this->extract_pred_from_proba(tr_probas);
            std::vector<int> pred_va = this->extract_pred_from_proba(va_probas);
            double metric_tr = metric->get(pred_tr, y_tr);
            double metric_va = metric->get(pred_va, y_va);
            this->print_epoch_log(epoch, metric_tr, metric_va, sum_residuals);
        }
    }
    delete pool;
}

std::vector<double> classic_classification::get_proba(const XY & d) const
{
    std::vector<double> odds (this->classes.size() * d.number_of_rows);
    for (int idx_row = 0; idx_row < d.number_of_rows; idx_row ++) 
    {
        for(long unsigned int idx_classe = 0; idx_classe < this->classes.size(); idx_classe++)
        {        
            odds[idx_row * this->classes.size() + idx_classe ] = this->log_odds_classes[idx_classe];
        }
    }
    for (int epoch = 0; epoch < conf::gbt::epochs; epoch ++) 
    {
        for (int idx_row = 0; idx_row < d.number_of_rows; idx_row ++) 
        {        
            for(long unsigned int idx_classe = 0; idx_classe < this->classes.size(); idx_classe++) 
            {    
                odds[idx_row * this->classes.size() + idx_classe ] += conf::gbt::learning_rate * this->trees[epoch * this->classes.size() + idx_classe]->predict_row<double>(d.x + idx_row * d.number_of_cols); 
            }
        }
    }
    return this->softmax(odds);
}

boost::python::numpy::ndarray classic_classification::predict_proba(const XY & d) 
{
    std::vector<double> probas = this->get_proba(d);
    int classes_numbers = this->classes.size();
    int row_numbers = probas.size() / classes_numbers;
        
    const double * data_ptr = probas.data();
    boost::python::tuple shape = boost::python::make_tuple(row_numbers, classes_numbers);
    boost::python::tuple stride = boost::python::make_tuple(sizeof(double) * classes_numbers, sizeof(double) );
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<double>();
    return boost::python::numpy::from_data(data_ptr, dt, shape, stride, boost::python::object()).copy();
}


boost::python::numpy::ndarray classic_classification::predict(const XY & d) 
{
    const std::vector<double> probas = this->get_proba(d);
    std::vector<int> preds = this->extract_pred_from_proba(probas);
    
    return boost::python::numpy::from_data(preds.data(),
        boost::python::numpy::dtype::get_builtin<int>(), 
        boost::python::make_tuple(preds.size()), 
        boost::python::make_tuple(sizeof(int)), 
        boost::python::object()).copy();
}

//////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Regression /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void regression::fit(XY & tr, const XY & va) 
{
    ThreadPool * pool = new ThreadPool(conf::number_of_threads);
    const double* y_tr = tr.get_y<double>();
    const double* y_va = va.get_y<double>();
    std::unique_ptr<base_factory> factory = base_factory::get_instance();  
    std::shared_ptr<base_criterion> criterion = factory->Criterion(); 
    std::shared_ptr<base_metrics> metric = factory->Metric(); 

    std::vector<double> pred_tr_final(tr.number_of_rows, 0.0);
    std::vector<double> pred_va_final(va.number_of_rows, 0.0);
    
    std::vector<double> tr_residuals;
    tr_residuals.insert(tr_residuals.end(), y_tr, y_tr + tr.number_of_rows); 
    
    for (int epoch = 1; epoch < conf::gbt::epochs + 1; epoch++)
    {        
        tree<double>* my_tree = new tree<double>(criterion, pool);
        
        my_tree->fit(tr, tr_residuals);
        this->trees.push_back(my_tree);
        
        for (int index_row = 0; index_row < va.number_of_rows; index_row ++)
        {
            pred_va_final[index_row] += conf::gbt::learning_rate * my_tree->predict_row<double>(va.x + index_row * va.number_of_cols);
        }
        
        double sum_residuals = 0;
        for (int index = 0; index < tr.number_of_rows; index ++)
        {
            double row_pred = conf::gbt::learning_rate * my_tree->predict_row<double>(tr.x + index * tr.number_of_cols);
            tr_residuals[index] -= row_pred;
            pred_tr_final[index] += row_pred; 
            sum_residuals += row_pred;
        }
        this->residuals_saved.push_back(sum_residuals);        

        if (conf::verbose == 1) 
        {
            double metric_tr = metric->get(pred_tr_final, y_tr);
            double metric_va = metric->get(pred_va_final, y_va);
            this->print_epoch_log(epoch,metric_tr, metric_va, sum_residuals );
        }

    }
    delete pool; 
}

boost::python::numpy::ndarray regression::predict(const XY & d) 
{
    std::vector<double> preds(d.number_of_rows, 0.0);
    for(auto const tree : trees) 
    {
        for (int index_row = 0; index_row < d.number_of_rows; index_row ++)
        {
            preds[index_row] += conf::gbt::learning_rate * tree->predict_row<double>(d.x + index_row * d.number_of_cols);
        }         
    }  
    return boost::python::numpy::from_data(preds.data(),
        boost::python::numpy::dtype::get_builtin<double>(), 
        boost::python::make_tuple(preds.size()), 
        boost::python::make_tuple(sizeof(double)), 
        boost::python::object()).copy();
}


