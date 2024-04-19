#include "Item.h"

Item::Item (int id, int width, int height): m_id(id), m_width(width), m_height(height){

}


Item::Item (int id, int width, int height, int x, int y, bool rotated) : m_id(id), m_width(width), m_height(height), m_x(x), m_y(y), m_rotated(rotated){

}


int Item::getId(){
  return m_id;
}

int Item::getWidth(){
  if (m_rotated){
    return m_height;
  }
  
  return m_width;
}

int Item::getHeight(){
  if (m_rotated){
    return m_width;
  }
  
  return m_height;
}

int Item::getX(){
  return m_x;
}

int Item::getY(){
  return m_y;
}

bool Item::isRotated(){
  return m_rotated;
}
