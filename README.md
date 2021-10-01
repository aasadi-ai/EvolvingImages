# Evolving Mickey Mouse:

This project contains the implementation of the Genetic Algorithm (GA) discussed in my article [Evolving Mickey Mouse](https://medium.com/@ahmedasadi.ai/evolving-mickey-mouse-c8542e4541c3).

### File Structure:
**EvolvingImage.py:** This file contains the class-based implementation of the GA, and takes as input the path to an image file and an optional set of hyperparameter values.

### Standard Usage:
```python
test = EvolvingImage("mickeyMouse.jpg")
test.n_Steps(5000)
test.displayRecord()
test.referenceImg()
test.topPerformerImg()
```
1. **EvolvingImage(imgPath)**: We pass the file path of the image we'd like to guess/search for into the EvolvingImage constructor.
2. **n_Steps(z)**: Evolves the population for z many generations
3. **displayRecord**: Uses matplotlib to chart the average fitness of a population at each generation
4. **referenceImg**: Displays the image we'd like to guess
5. **topPerformerImg**: Displays the image of the fittest individual in our current population
