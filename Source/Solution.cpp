#include "Solution.h"
#include <set>

Solution::Solution(int binWidth, int binHeight): m_binWidth(binWidth), m_binHeight(binHeight){
  m_items = std::list<std::pair<int, Item>>();
}


int Solution::getFitness(){

}

int Solution::getNbBins(){
  std::set<int> binsId;

  for (auto it = m_items.begin(); it != m_items.end(); ++it){
    binsId.insert(it->first);
  }

  return binsId.size();
}

int Solution::getTheoricalMininalNbBins(){
  int totalArea = 0;
  for (auto it = m_items.begin(); it != m_items.end(); ++it){
    totalArea += it->second.getArea();
  }

  return totalArea / (m_binWidth * m_binHeight);
}


void Solution::addItemsInNewBin(Item item){
  m_items.push_back(std::pair<int, Item>(m_items.size(), item));
}
