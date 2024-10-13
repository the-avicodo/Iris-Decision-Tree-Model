# Iris Classification Model (Decision Tree Model)

### Table of Contents
1. Project Overview
2. Data Sources
3. Tools
4. Data Cleaning and Prep
5. Data Analysis
6. Results
7. Limitations

### Project Overview
This data analysis project aims to build a model capable of accurately classifying different Iris flower species based on 4 Iris flower measurements:
1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

I created this project as an introduction to creating classification models in Python.
  
### Data Sources
To train the model, I downloaded the Iris dataset from [Kaggle.com](https://www.kaggle.com/datasets/uciml/iris/data).

### Tools
- Python (for data cleaning and analysis)
  - Numpy, Pandas, and Scikit-learn

### Data Cleaning and Prep
In the initial cleaning and prep, I performed the following steps:
1. Loaded data into Python notebook with pandas library
2. Inspected data for missing values and duplicates
3. Performed "One-Hot Encoding" on "Species" column
   - 0 = Setosa
   - 1 = Versicolor
   - 2 = Virginica
     
### Data Analysis
After I cleaned and prepped the data, I created a decision tree classifier with help from a YouTube tutorial by [Normalized Nerd](https://www.youtube.com/watch?v=sgQAhG5Q7iY&t=952s) using multiple methods and class variables. This was later used to develop and test the decision tree model.
```python
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        '''constructor'''
        
        #Initialize root of tree
        self.root = None

        #stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        '''recursive function to build tree'''

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        #split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth <= self.max_depth:
            #find best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            #check if info gain si positive
            if best_split["info_gain"] > 0:
                #recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                #recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                #return decision node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree,
                            right_subtree, best_split['info_gain'])
```
To see the full code, open the ".ipynb" file in this repository.

### Results
The model performed exceptionally with an accuracy score of 93%.

### Limitations
This dataset has only 150 data points (50 data points per class), which is a relatively small dataset size. Because of this, certain complex relationships are unable to be captured. Due to the small dataset size, a decision tree model works well.

