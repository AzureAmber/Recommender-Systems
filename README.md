# Recommender-Systems

In this project, I will be showcasing how to implement various
types of recommendation systems including:

- Popularity Matching
- Content-based Filtering
- Collaborative Filtering

Additionally, I will also implement a bit of Natural Language
Processing (NLP) in order to perform predictive analysis.

In this project, I will provide explanations and implemntations in Python.





## A: Data

I will be using two datasets which can be found in `data` directory.
The first dataset `Restaurants` contains basic information about restaurants
in Evanton, IL. The second dataset `Reviews` contains reviews of the restaurants
along with some information on the reviewers.

The `Restaurants` dataset is already clean so no issues needs to be addressed there.
However, the `Reviews` dataset has some basic issues (missingness) that needs to be
prepared properly in order to be usable. To address missingness, I will be using
different imputation methods:

- I will be using `Review Text` for NLP later so impute missingness with any empty string
- For categorical variables (`Vegetarian?`, `Marital Status`, `Has Children?`), impute missingness as an additional class label
- Since other variables have few missingess, just impute with the mean, median, or mode
- There are no numerical variables with severe missingness, but if they exist, a good method to perform imputation is to utilize clustering techniques (K-means, Gaussian Mixture Models, DBSCAN, Expectation-Maximization, etc). This is standard practice, but I will not be covering these techniques in this paper as they are not the main focus.

You can locate my data preparation in the `code` directory titled `data_preparation`.



## B. Implementing Recommender Systems

### B1. Popularity Matching

A Popularity Matching System is pretty simple. Just recommend to user the restaurants that
are the most popular by some metric.

In this case, I will be using the `Rating` parameter as the metric in the `Reviews` dataset.

One method to aggregate multiple ratings per restaurant is to find the average rating for each 
restaurant. For n restarants, let $\mu_i$ be the true rating of the i-th restaurant and
$n_i$ be the number of ratings for the i-th restaurant. If $r_{ij}$ is the j-th rating for
the i-th restaurant. Then, $\mu_i$ can be approximated by the mean of $r_{ij}$.

$$
    \mu_i \approx \overline{r_i}
$$

```py
reviews.groupby(['Restaurant Name']).agg(count=('Rating', 'size'), avg_Rating=('Rating', 'mean')).reset_index()
```

Another method is to use some kind of shrinkage estimator. Restaurants with
more reviews can approximate $\mu_i$ more accurately while less reviews are less
accurately. Let $\mu$ be the true average rating for restaurants which can be estimated
by $\overline{r}$.

$$
    \mu^*_i \approx \frac{n_i}{n_i+\overline{n_i}} \overline{r_i}
        + \frac{\overline{n_i}}{n_i+\overline{n_i}} \overline{r}
$$

Thus, restaurants with more reviews will have a shrinkage rating that is closer
to its actual average while less reviews causes a shrinkage rating that is closer
to the true average rating for restaurants.

```py
rating_mean = reviews['Rating'].mean()

rating_count_mean = avg_scores['count'].mean()

avg_scores['shrinkage_Rating'] = (avg_scores['count'] * avg_scores['avg_Rating'] + rating_count_mean * rating_mean) / (avg_scores['count'] + rating_count_mean)
```

Finally, using either the average rating or shrinkage rating, recommend restaurants
with the highest ratings. For my implementation, I also recommend restuarants based
on the tpye of cuisine.

```py
def popularity_recommendation(cuisine_type, score_type='Default'):
    if (score_type == 'Shrinkage'):
        res_list = data_res_scores[data_res_scores['Cuisine'] == cuisine_type].sort_values(['shrinkage_Rating', 'count'], ascending=[False,False])

        return(res_list.loc[:,['Restaurant Name', 'count', 'avg_Rating', 'shrinkage_Rating']].reset_index(drop=True))
    else:
        res_list = data_res_scores[data_res_scores['Cuisine'] == cuisine_type].sort_values(['avg_Rating', 'count'], ascending=[False,False])
        
        return(res_list.loc[:,['Restaurant Name', 'count', 'avg_Rating', 'shrinkage_Rating']].reset_index(drop=True))
```








