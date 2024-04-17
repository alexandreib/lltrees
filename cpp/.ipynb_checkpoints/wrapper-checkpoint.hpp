#pragma once
#include <boost/python/numpy.hpp>
#include <iostream>
#include <numeric>

class XY 
{
public:
~XY() = default;
double* x = NULL;
int size_x, number_of_rows, number_of_cols;
std::vector<int> index;
int number_of_classes;

// XY
void set_x(const boost::python::numpy::ndarray & np_x);
std::vector<double> get_column(const int index_col);
std::vector<double> get_column(const int index_col, const std::vector<int>& index) const;

const std::vector<double> & get_vector_proba() const;

template<class T>
T* get_y() const;

template<class T>
void set_pred(std::vector<T> preds);

virtual void set_y(const boost::python::numpy::ndarray & np_y) = 0;
virtual void set_xy(const boost::python::numpy::ndarray & np_x, const boost::python::numpy::ndarray & np_y) = 0;
};

template<class T>
class Y : public XY 
{
public :
Y();
Y(const boost::python::numpy::ndarray & np_x);
Y(const boost::python::numpy::ndarray & np_x, const boost::python::numpy::ndarray & np_y);
~Y() ;
T* y = NULL;
std::vector<T> pred; 
// for regression : pred => double
// for classification : pred => int
// also for classification : using derived class, with pred double for probability

void set_y(const boost::python::numpy::ndarray & np_y) override;
void set_xy(const boost::python::numpy::ndarray & np_x, const boost::python::numpy::ndarray & np_y) override;
};
