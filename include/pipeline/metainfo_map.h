#pragma once
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

class MetaInfoMap {
public:
  MetaInfoMap();
  ~MetaInfoMap();

  int size();
  void set(const std::string &key, std::map<std::string, void *> &value);
  std::map<std::string, void *> get(const std::string &key);
  void process(const std::string &key,
               void(process_fun)(std::map<std::string, void *> &,
                                 const std::vector<void *> &),
               const std::vector<void *> &param);
  void erase(const std::string &key);

private:
  mutable std::shared_timed_mutex _mutex;
  std::map<std::string, std::map<std::string, void *>> _map;
};