#pragma once

#include <filesystem>
#include <functional>
#include <string>

class DataLoader {
public:
    std::function<void(std::string)> log_fn;
    std::filesystem::path load_dir;

    DataLoader(const std::filesystem::path& load_dir) : load_dir(load_dir) {};

    void load_data();
};

