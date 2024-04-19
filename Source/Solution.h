#include "Item.h"
#include <list>

class Solution{

  public :
    Solution ();

    int getFitness();
    int getNbBins();


    void addItemsInNewBin(Item item);



    /*
    OPERATION DE VOISINAGE
    */

    // return SOLUTION_NULL if the rotated item can't fit or it's a sqare
    Solution rotateItem(int itemId);

    // move the item to the new position inside the same bin
    // return SOLUTION_NULL if the item can't move to the new position
    //ATTENTION: guillotine
    Solution moveItem(int itemId, int x, int y);

    Solution exchangeItemsBetweenBin(int itemId1, int itemId2);

    Solution moveItemToAnotherBin(int itemId, int binId);


  protected :

    // <binId, item>;
    std::list<std::pair<int, Item>> m_items;
  
};
