#include <iostream>
#include <filesystem>

#include "data_loader.h"
#include "utils.h"

void DataLoader::load_data() {
    utils::CppLogger::info(load_dir);
    // std::cout << "load dir: " << load_dir << std::endl;
}

