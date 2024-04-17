#include "wrapper.hpp"


//////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// XY ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
template <class T> 
T* XY::get_y() const
{
    const Y<T>* subclass = static_cast<const Y<T>*>(this);
    return subclass->y;
}

template int* XY::get_y() const; // explicit instantiation.
template double* XY::get_y() const; // explicit instantiation.

template <class T> 
void XY::set_pred(std::vector<T> preds)
{
    Y<T>* subclass = static_cast<Y<T>*>(this);
    subclass->pred = preds;
}
template void XY::set_pred(std::vector<int>); // explicit instantiation.
template void XY::set_pred(std::vector<double>); // explicit instantiation.

const std::vector<double> & XY::get_vector_proba() const
{
    const Y<double> * subclass = static_cast<const Y<double>*>(this);
    return subclass->pred;
}

void XY::set_x(const boost::python::numpy::ndarray & np_x)
{
    this->number_of_rows = np_x.shape(0);
    this->number_of_cols = np_x.shape(1);
    this->size_x = np_x.shape(0) * np_x.shape(1);
    this->x = reinterpret_cast<double *>(np_x.get_data()); 
}

std::vector<double> XY::get_column(const int index_col)
{
    std::vector<int> index(this->number_of_rows);
    std::iota(index.begin(), index.end(), 0);
    this->index = index;
    return this->get_column(index_col, index);
}     

std::vector<double> XY::get_column(const int index_col, const std::vector<int>& index) const 
{
    std::vector<double> columns;
    for(auto const &index_row : index) {
        columns.push_back(this->x[index_row * this->number_of_cols + index_col]); 
    }
    return columns;
} 

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Y /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
template <class T> 
Y<T>::Y()
{
}

template <class T> 
Y<T>::Y(const boost::python::numpy::ndarray & np_x)
{
    this->set_x(np_x);
}

template <class T> 
Y<T>::Y(const boost::python::numpy::ndarray & np_x, const boost::python::numpy::ndarray & np_y)
{
    this->set_xy(np_x, np_y);
}


template <class T> 
Y<T>::~Y()
{
    delete this->x;
    delete this->y;
    std::cout<<"delete"<<std::endl;
}

template <class T> 
void Y<T>::set_y(const boost::python::numpy::ndarray & np_y)
{
    this->y = reinterpret_cast<T *>(np_y.get_data()); 
}

template <class T> 
void Y<T>::set_xy(const boost::python::numpy::ndarray & np_x, const boost::python::numpy::ndarray & np_y)
{
    this->set_x(np_x);
    this->set_y(np_y);
}

template class Y<int>;// Explicit instantiation
template class Y<double>;// Explicit instantiation
