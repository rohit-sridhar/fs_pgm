#include <iostream>
#include <filesystem>
#include <string>

#include "data_loader.h"
#include "utils.h"

void DataLoader::load_data() {
    std::string str1 = "TEST MESSAGE";
    utils::CppLogger::log_info() << str1;
    // std::cout << "load dir: " << load_dir << std::endl;
}

