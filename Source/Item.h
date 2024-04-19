
class Item{

  public :
    Item (int id, int width, int height);
    Item (int id, int width, int height, int x, int y, bool rotated);

    int getId();
    int getWidth();
    int getHeight();
    int getX();
    int getY();
    bool isRotated();

    int getXTopLeft();
    //... des fcns pour obtenir les coordonnées des 4 coins de l'item

  protected :

    int m_id;
    int m_width;
    int m_height;
    int m_x;
    int m_y;
    bool m_rotated;
    
};
