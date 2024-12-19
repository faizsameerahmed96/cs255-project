# TSP with Genetic Algorithm
This is a comparative study between Held karp and Genetic algorithm for TSP. 

# How to run the file
## Running with random assignment of cities positions

We can run the analysis for random assignment of cities by running:
- `python algorithm_analyzer.py`
- When prompted to enter number of cities, enter the number.

  ```
  Enter the number of cities: 
  10
  ```
- The result is stored under `result/{latest_time_stamp}`

## Running with custom cities coordinates
- `python algorithm_analyzer.py --file ./custom_coordiantes.txt`
- Please see `custom_coordiantes.txt` for format of coordinates file.
- The result is stored under `result/{latest_time_stamp}`



## Slides Action Items
- [x] Create a way to specify custom coordinates (test case), text file
- [x] Show test case and output in some file
- [x] Save output in output files.
- [x] Give option to input test case, save test case and output. If user does not provide test case, randomly generate. User specifies number of cities.
- [ ] More test cases for number of cities (2 per number of cities)
- [ ] Compare genetic and held karper
- [ ] Add more description on genetic algorithm, (parameters), held karper
- [ ] Runtime increase of genetic algorithm and solution quality, compare with held karper (upto 20)
- [ ] Remove other slides, make it simple
