# 2D Bin Packing Problem

The 2D Bin Packing Problem involves efficiently packing a set of rectangular items into a limited number of fixed-size bins while optimizing certain criteria, such as maximizing space utilization or minimizing the number of bins used.

In this project, We propose a **hybrid evolutionary algorithm** and a **tabu search algorithm** for solving the 2D bin packing problem in which the items can be rotated and must follow the guillotine cut property.

## Versions of the 2D Bin Packing Problem

The 2D Bin Packing Problem can be categorized based on whether the items can be rotated and whether the items must adhere to the guillotine cut property. Each version addresses different constraints:

- **Rotation (0 or R)**:
  - When rotation is not allowed (0), items must be placed in their original orientation.
  - When rotation is allowed (R), items can be rotated to better fit within the bin, which can lead to higher space utilization.

- **Guillotine Cut Property (F or G)**:
  - Guillotine free (F) allows for more flexibility in item placement as cuts don't need to be considered.
  - Guillotine cut property (G) requires that items be placed in a manner that allows for them to be separated from the bin using a series of edge-to-edge cuts. This mimics real-world cutting processes and often adds complexity to the problem.

By considering these variations, we can tackle different scenarios of the 2D Bin Packing Problem:

- **2DBP | 0 | F**: Simplest form where items are placed in their original orientation without the need for guillotine cuts.
- **2DBP | 0 | G**: Items must be placed in their original orientation and follow the guillotine cut property.
- **2DBP | R | F**: Items can be rotated, offering flexibility, without needing to follow the guillotine cut property.
- **2DBP | R | G**: Most complex form where items can be rotated and must follow the guillotine cut property.

## Implementation

### Model
We use the **Largest-Gap-Fit Increasing (LGFI) algorithm** to place items into bins. This algorithm efficiently finds the largest available gap within the bin to fit each item, ensuring optimal space utilization.

**Solution Representation :**
In our approach, a solution is represented by the specific order in which items are placed using the LGFI algorithm. The order is crucial as it directly influences the efficiency of space utilization within each bin.

**Item Orientation :**
Each item in the solution can be in one of two orientations :
- Normal: The item is placed with its original width and height.
- Rotated: The item's width and height are reversed.

### Fitness Function
Our fitness function is formulated to address the dual objectives of reducing the number of bins and maximizing the space utilization within the last bin.
$$ F = Nbins​ + \frac{1−LastBinArea}{\sum ItemsAreaInTheLastBin​} $$

Both our hybrid evolutionary algorithm and tabu search algorithm are designed to minimize this fitness function.

### Hybrid Genetic Algorithm

We implemented a **Hybrid Genetic Algorithm** proposed in the paper *"A Hybrid Genetic Algorithm for the 2D Guillotine Cutting Problem"* ([original paper](https://www.sciencedirect.com/science/article/pii/S1877050913003980?ref=pdf_download&fr=RR-2&rr=886d22872d9e189c)), to which we added new features to handle rotation and the guillotine cut property, as well as mutations.

To tackle the most complex version, **2DBP | R | G** (Rotated and Guillotine Cut Property), we implemented our solution in **Python** using **Numba** to optimize performance.

### Tabu Search Algorithm

We also implemented a classic **Tabu Search Algorithm** with 3 kinds of neighbourhood :
- **rotation :** rotate 1 item (n neighbours)
- **permutation :** swap 2 side-by-side items (n-1 neighbours)
- **insertion :** insert 1 item to the first position and shift the others (n-1 neighbours)

### Optimizing with Numba

**Numba** is a **Just-In-Time (JIT)** compiler for Python that translates a subset of Python and NumPy code into **fast machine code**. This significantly **improves the performance of numerical computations**. It allows developers to write code in Python while achieving performance close to that of lower-level languages.

To further enhance performance, we use **Numba's caching technique**, which **avoids recompilation** at every execution by storing the compiled machine code. This means that the first time a function runs, Numba compiles and caches it, and subsequent executions retrieve the compiled code from the cache, **reducing overhead and improving runtime efficiency**.

## How to Launch

First, ensure all the required packages are installed by executing this command: `pip install -r requirements.txt`

To run the program, go to `main.py` and unselect the parts of the code based on your needs:

1. **Compile Code**: Select this to compile all necessary functions using Numba. 
   - You can set the `advanced` flag to **True** to get more insights into the compilation time of each function.
2. **Generate Solutions**: Select this to run the genetic algorithm on each input file.
3. **Visualize Solutions**: Select this to visualize the solution of a specific packing scenario.

## Results

Here are examples of solutions generated by the genetic algorithm for the **2DBP | R | G** scenario (items can be rotated and must follow the guillotine cut property):

![Example Solution 1](img/solution_08.png)
*Example Solution 8: Items are packed using rotations and guillotine cuts to maximize space utilization.*

![Example Solution 2](img/solution_13.png)
*Example Solution 13: Another scenario showing efficient packing with rotations and guillotine cuts.*

These examples illustrate how our hybrid genetic algorithm effectively optimizes space utilization and adheres to the guillotine cut property.

### Lower bound
Considering a set of items and a particular size of bin, if with assume that it is possible to perfectly fit all the items into bins, the minimal number of bin will be :
$$ minimalBinNumber = \frac{\sum itemArea}{binArea} $$

For our dataset, here are the minimal number of bins according to this formula : 
|file name|minimal number of bins|bin area|items area|
|----------------------|--------|---------|----------|
| binpacking2d-01.bp2d | 3      | 62500   | 163562   |
| binpacking2d-02.bp2d | 5      | 62500   | 274563   |
| binpacking2d-03.bp2d | 7      | 62500   | 407651   |
| binpacking2d-04.bp2d | 12     | 62500   | 731408   |
| binpacking2d-05.bp2d | 3      | 250000  | 545300   |
| binpacking2d-06.bp2d | 5      | 250000  | 1232057  |
| binpacking2d-07.bp2d | 9      | 250000  | 2004791  |
| binpacking2d-08.bp2d | 12     | 250000  | 2805462  |
| binpacking2d-09.bp2d | 3      | 1000000 | 2021830  |
| binpacking2d-10.bp2d | 6      | 1000000 | 5355377  |
| binpacking2d-11.bp2d | 7      | 1000000 | 6536520  |
| binpacking2d-12.bp2d | 13     | 1000000 | 12521992 |


