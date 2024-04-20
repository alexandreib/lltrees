#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <set> 
#include <vector>
#include <cmath>
#include "wrapper.hpp"
#include "tree.hpp"


class base_gbt 
{
public:
inline void print_epoch_log(int & epoch, double & metric_tr, double & metric_va, double residuals = 0.0 );

virtual ~base_gbt() = default;
virtual void print() = 0;
virtual void save() = 0;
virtual void load() = 0;
virtual void fit(XY & tr, const XY & va) = 0;
virtual boost::python::numpy::ndarray predict(const XY & ts) = 0;

virtual boost::python::numpy::ndarray predict_proba(const XY & d) {std::cout <<  __PRETTY_FUNCTION__ << std::endl; __builtin_unreachable(); };

virtual std::vector<double> get_residuals() const {std::cout <<  __PRETTY_FUNCTION__ << std::endl; __builtin_unreachable(); };

virtual std::vector<double> get_proba(const XY & d) const {std::cout <<  __PRETTY_FUNCTION__ << std::endl; __builtin_unreachable(); };

virtual std::vector<int> get_predict(const XY & d) const  {std::cout <<  __PRETTY_FUNCTION__ << std::endl; __builtin_unreachable(); };

};

template<class T> 
class gbt : public base_gbt 
{
protected:
std::vector<tree<T>*> trees;
public: 
gbt();
~gbt();
void print() override;
void save() override;
void save(std::ofstream & file);
void load() override; 
void load(std::ifstream & file);
};

class abstract_classification  
{
protected:
std::set<int> classes; 
public: 
inline std::vector<double> softmax(std::vector<double> odds) const;
inline std::vector<int> extract_pred_from_proba(const std::vector<double> probas) const;
};

class abstract_classic_boost : public gbt<double>
{
protected:
std::vector<double> residuals_saved; 
public: 
std::vector<double> get_residuals() const override;
};

class regression : public abstract_classic_boost
{
private:
public: 
void fit(XY & tr, const XY & va) override;
boost::python::numpy::ndarray predict(const XY & ts) override;
};

class adaboost_classification : public gbt<int>, public abstract_classification
{
private:
std::vector<double> models_weights;

public: 
void save() override;
void load() override;
void fit(XY & tr, const XY & va) override;
boost::python::numpy::ndarray predict(const XY & ts) override;
boost::python::numpy::ndarray predict_proba(const XY & d) override;
std::vector<int> get_predict(const XY & d) const override;
std::vector<double>  get_proba(const XY & d) const override;

};

class classic_classification : public abstract_classification, public abstract_classic_boost
{
private:
std::vector<double> log_odds_classes; 
public :
void save() override;
void load() override;
void fit(XY & tr, const XY & va) override;
boost::python::numpy::ndarray predict(const XY & d) override;
boost::python::numpy::ndarray predict_proba(const XY & d) override;
std::vector<double>  get_proba(const XY & d) const override;
};