#include "pipeline/metainfo_map.h"
#include <mutex>
#include <iostream>

MetaInfoMap::MetaInfoMap() {}

MetaInfoMap::~MetaInfoMap() {}

int MetaInfoMap::size() {
  std::shared_lock<std::shared_timed_mutex> lock(_mutex);
  return _map.size();
}

void MetaInfoMap::set(const std::string &key,
                      std::map<std::string, void *> &value) {
  // std::unique_lock<std::shared_timed_mutex> lock(_mutex);
  std::lock_guard<std::shared_timed_mutex> lock(_mutex);
  _map[key] = value;
}

std::map<std::string, void *> MetaInfoMap::get(const std::string &key) {
  std::shared_lock<std::shared_timed_mutex> lock(_mutex);
  auto iter = _map.find(key);
  if (iter != _map.end()) {
    return iter->second;
  }
  return std::map<std::string, void *>();
}

void MetaInfoMap::process(const std::string &key,
                          void(process_fun)(std::map<std::string, void *> &,
                                            const std::vector<void*> &),
                          const std::vector<void*> &param) {
  std::shared_lock<std::shared_timed_mutex> lock(_mutex);
  auto iter = _map.find(key);
  if (iter != _map.end()) {
    process_fun(iter->second, param);
  }
}

void MetaInfoMap::erase(const std::string &key) {
  // std::unique_lock<std::shared_timed_mutex> lock(_mutex);
  std::lock_guard<std::shared_timed_mutex> lock(_mutex);
  _map.erase(key);
}