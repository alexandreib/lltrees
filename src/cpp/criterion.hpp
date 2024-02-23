#ifndef __CRITERIONE_H_INCLUDED__
#define __CRITERIONE_H_INCLUDED__

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cassert>

class criterion{
private:
    std::string criterion_name;

    double variance(const std::vector<double>& Y) {
        double average = 0, sum = 0;
        for(double value : Y){average = average + value;}
        average = average / Y.size();
        for(double value : Y) {sum += pow(average - value,2);} 
        return sum;  
    }

public:
    criterion() : criterion_name("variance") {};
    criterion(std::string criterion_name) : criterion_name(criterion_name) {};

    void set_name(std::string criterion_name) {
        this->criterion_name = criterion_name;
        std::cout<<"set criterion_name : " << this->criterion_name <<std::endl;
    }

    std::string get_name() {
        std::cout<<"get criterion_name : " << this->criterion_name <<std::endl;
        return this->criterion_name;
    }

    void print() {
        std::cout << "Critetion available for decisison tree : variance." << std::endl;
    }

    double get(const std::vector<double>& Y){
        if (this->criterion_name == "variance") {
            return this->variance(Y);
        }
        else {
            assert(true && "criterion_name is not defined");
            return 0; // to avoid the warning during compile
        }
    }


};

#endif // __CRITERIONE_H_INCLUDED__