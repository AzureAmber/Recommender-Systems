# Recommender-Systems

Many systems today have implemented ways to generate useful or liked
suggestions for users to engage with (i.e. Netflix, Youtube, Spotify, etc).
Each of them use some type of recommender system, an information filtering
system that generates suggestions.
In this project, I will be showcasing how to implement a few
types of recommendation systems including:

- Popularity Matching
- Content-based Filtering
    - Using numerical / categorical data
    - Using simple Natural Language Processing (NLP) for textual data
- Collaborative Filtering
    - Using information about users
    - Using information about users related to the output

In this project, I will be explaining how to implement the recommender systems above
in Python using recommendations for various restaurants around Evanston, IL.





## A: Data

I will be using two datasets which can be found in `data` directory.
The first dataset `Restaurants` contains basic information about restaurants
in Evanton, IL. The second dataset `Reviews` contains reviews of the restaurants
along with some information on the reviewers.

Remember, the first step to any data project is to prepare and 'fix' the data.
The `Restaurants` dataset is already clean so no issues needs to be addressed there.
However, the `Reviews` dataset has some basic issues (missingness) that needs to be
prepared properly in order to be usable. To address missingness, I will be using
different imputation methods:

- I will be using `Review Text` for NLP later so impute missingness with any empty string
- For categorical variables (`Vegetarian?`, `Marital Status`, `Has Children?`), impute missingness as an additional class label
- Since other variables have few missingess, just impute with the mean, median, or mode
- There are no numerical variables with severe missingness, but if they exist, a good method to perform imputation is to utilize clustering techniques (K-means, Gaussian Mixture Models, DBSCAN, Expectation-Maximization, etc). This is standard practice, but I will not be covering these techniques in this paper as they are not the main focus.

You can locate my data preparation in the `code` directory titled `data_preparation`.





## B. Popularity Matching

A Popularity Matching System is pretty simple. Just recommend the most popular item
through some metric.

In this case, I will be using the `Rating` parameter as the metric in the `Reviews` dataset
to recommend restaurants.

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
def popularity_recommendation(cuisine_type, max_sugest, score_type='Default'):
    if (score_type == 'Shrinkage'):
        res_list = data_res_scores[data_res_scores['Cuisine'] == cuisine_type].sort_values(['shrinkage_Rating', 'count'], ascending=[False,False])
        return(res_list.loc[:,['Restaurant Name', 'count', 'avg_Rating', 'shrinkage_Rating']].head(max_sugest).reset_index(drop=True))
    else:
        res_list = data_res_scores[data_res_scores['Cuisine'] == cuisine_type].sort_values(['avg_Rating', 'count'], ascending=[False,False])
        return(res_list.loc[:,['Restaurant Name', 'count', 'avg_Rating', 'shrinkage_Rating']].head(max_sugest).reset_index(drop=True))
```

You can locate my Popularity Matching Recommendation System in the `code` directory titled `popularity`.



## C. Content-based Filtering

### C1. Using numerical and categorical information

A Content-based Filtering System uses information about each item and recommends items
that are the most similar through some metric. Some common similarity metrics are:

- Euclidean distance: $|x-y|^2$ for $x, y \in \mathbb{R}^n$
- Cosine distance: $\frac{x \cdot y}{|x| |y|}$ for $x, y \in \mathbb{R}^n$

There are two main issues with these metrics.

(1) For numerical variables, their values can be in different scales
(i.e. \\$, $\degree C$, mi, etc).
A common solution is to apply some kind of normalization. Some types are:

- Standard Norm: $\frac{x - \overline{x}}{\sigma_x}$
- Linear Norm: $\frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}$

(2) Categorical variables cannot be directly used. You need to apply some kind of
numerical encoding to transform categorical into numerical.
A common method is to use one-hot encoding.
Basically, if a categorical variable $C$ has m possible classes, $c_1,...,c_m$, create m-1 new
variables, $C_1,...,C_{m-1}$, where the $C_i = 1$ if the $C = c_i$ and
$C_i = 0$ if the $C \not = c_i$.

```py
# one-hot encoding for categorical
data_cat = restaurants[['Cuisine', 'Open After 8pm?']]
data_cat = pd.get_dummies(data_cat, columns=['Cuisine', 'Open After 8pm?'], drop_first=True, dtype=int)
# standardization for numerical
data_num = restaurants[['Latitude', 'Longitude', 'Average Cost']]
scaler = StandardScaler()
data_num = scaler.fit_transform(data_num)

data_restaurant = np.column_stack((data_num, data_cat))



# euclidean distance
restaurants_euclidean = pd.DataFrame(
    euclidean_distances(data_restaurant, data_restaurant),
    columns=restaurants['Restaurant Name'],
    index=restaurants['Restaurant Name']
)
# cosine distance
restaurants_cosine = pd.DataFrame(
    cosine_distances(data_restaurant, data_restaurant),
    columns=restaurants['Restaurant Name'],
    index=restaurants['Restaurant Name']
)
```

Finally, recommend restaurants with the high similarity to a restaurant that the user
highly rated. For my implementation, I recommended restaurants that are similar
to the user's most popular restaurant.

```py
def contentfilter_recommendation(name, score_type='Euclidean'):
    reviewer_restaurants = reviews[reviews['Reviewer Name'] == name].sort_values(['Rating'], ascending=False)
    fav_restaurant = reviewer_restaurants['Restaurant Name'].iloc[0]
    if (score_type == 'Cosine'):
        data_ret = restaurants_cosine.loc[:,fav_restaurant].sort_values(ascending=True)
        return( data_ret[data_ret.index != fav_restaurant].head(10) )
    else:
        data_ret = restaurants_euclidean.loc[:,fav_restaurant].sort_values(ascending=True)
        return( data_ret[data_ret.index != fav_restaurant].head(10) )
```

You can locate my Content-based Filtering Recommendation System in the `code` directory titled `content_based1`.

### C2. Using NLP

Another method instead of using numerical or categorical variables is to use NLP.
Since items sometimes are associated with some kind of description, you can use NLP
to convert descriptions into numerical embedding. Then, these embeddings can be used
with the similarity metric.

Some models that perform NLP well are BERT and GPT.
In this project, I will instead perform some low level NLP that are easier to understand.

(1) Jaccard Distance

Let two items have descriptions $D_i, D_j$. The Jaccard distance metric is the
proportion of number of unique words that the items have in common.

$$
    Jaccard(D_i,D_j) = \frac{D_i \cap D_j}{D_i \cup D_j}
$$

```py
restaurants_jaccard = np.zeros((len(restaurants),len(restaurants)))

words_per_restaurant = [set(filter(None, re.split('[.!?,; ]', x.lower()))) for x in restaurants['Augmented Description']]
for i in range(0,len(restaurants)):
    for j in range(0,len(restaurants)):
        same_len = len(words_per_restaurant[i].intersection(words_per_restaurant[j]))
        total_len = len(words_per_restaurant[i].union(words_per_restaurant[j]))
        restaurants_jaccard[i,j] = same_len / total_len
        
restaurants_jaccard = pd.DataFrame(
    restaurants_jaccard,
    columns=restaurants['Restaurant Name'],
    index=restaurants['Restaurant Name']
)
```

Then, recommend items with low Jaccard distances to the user's highly rated restaurant.
In my implementation, instead of choosing the user's highest rating restaurant, I
took a random restaurant that the user highly rated.

```py
def contentfilter_recommendation_jaccard(name, max_suggest):
    reviewer_restaurants = reviews[reviews['Reviewer Name'] == name]
    fav_restaurants = list(reviewer_restaurants[reviewer_restaurants['Rating'] == reviewer_restaurants['Rating'].max()]['Restaurant Name'])
    restaurant_to_input = random.sample(fav_restaurants, 1)[0]
    data_ret = restaurants_jaccard.loc[:,restaurant_to_input].sort_values(ascending=False)
    
    return( data_ret[data_ret.index != restaurant_to_input].head(max_suggest) )
```

(2) TF-IDF

One issue with Jaccard distance is that each word in the description has the same value.
But, words like 'the', 'is', etc hold little value. TF-IDF assigns values to words
with the number of times the word appears in the description.
For word w:


$$
    TF\_IDF(w,D_i) = \frac{\text{number of times w appears in } D_i}
                    {\text{number of words in } D_i}
        * \ln( \frac{\text{number of } D_i}{\text{number of } D_i \text{ with w}} )
$$

For my implementation, I took the top 100 words with the most occurence across
all descriptions. Then, for each restaurant's description,
calculate the TF-IDF for each of the 100 words.

```py
# function to calculate TF-IDF
def TD_IDF(word):
    word_count_by_restaurant = restaurants['Augmented Description'].apply(
        lambda x: list(filter(None, re.split('[.!?,; ]', x.lower()))).count(word.lower())
    )
    total_words_by_restaurant = restaurants['Augmented Description'].apply(
        lambda x: len(list(filter(None, re.split('[.!?,; ]', x.lower()))))
    )
    total_doc = len(restaurants['Augmented Description'])
    docs_with_word = sum(word_count_by_restaurant >= 1)
    
    restaurants_tdidf = pd.DataFrame(
        list(word_count_by_restaurant / total_words_by_restaurant * np.log(total_doc / docs_with_word)),
        columns=[word.lower()], index=restaurants['Restaurant Name']
    )
    
    return( restaurants_tdidf )

# top 100 most common words
descriptions_all_words = re.split('[.!?,; ]', " ".join(list(restaurants['Augmented Description'])).lower())
most_pop_words = list(pd.Series(filter(None, descriptions_all_words)).value_counts().sort_values(ascending=False).head(100).index)

# calculate tf-idf for each word
restaurants_td_idf = np.zeros((len(restaurants),len(most_pop_words)))

for j in range(0,len(most_pop_words)):
    cur_word_td_idf = TD_IDF(most_pop_words[j])
    for i in range(0,len(restaurants)): 
        restaurants_td_idf[i,j] = cur_word_td_idf.iloc[i]
        
restaurants_td_idf = pd.DataFrame(
    restaurants_td_idf,
    columns=most_pop_words,
    index=restaurants['Restaurant Name']
)
```

Finally, use the TF-IDF along with the euclidean distance similarity distance to
recommend similar restaurants.

```py
# calculate euclidean distances
restaurants_td_idf_euclidean = pd.DataFrame(
    euclidean_distances(restaurants_td_idf, restaurants_td_idf),
    columns=restaurants['Restaurant Name'], index=restaurants['Restaurant Name']
)
# recommendation system
def contentfilter_recommendation_td_idf(name, max_suggest):
    reviewer_restaurants = reviews[reviews['Reviewer Name'] == name]
    fav_restaurants = list(reviewer_restaurants[reviewer_restaurants['Rating'] == reviewer_restaurants['Rating'].max()]['Restaurant Name'])
    restaurant_to_input = random.sample(fav_restaurants, 1)[0]
    data_ret = restaurants_td_idf_euclidean.loc[:,restaurant_to_input].sort_values(ascending=True)
    
    return( data_ret[data_ret.index != restaurant_to_input].head(max_suggest) )
```

You can locate my Content-based Filtering Recommendation System with NLP in the `code` directory titled `content_based2`.



## D. Collaborative Filtering

A Collaborative Filtering System recommends items indirectly by finding similar users.

In this case, find someone that is similar to the user and recommend a restaurant that he/she
highly rated. Thus, the main objective of a Collaborative Filtering System is to find
similar users.

### D1. Using information about the users

Use information about the users to find similar users and then recommend items
from similar users. Note this is similar to the Content-based Filtering System, but the 
information used is the user data instead of the data about the items (i.e the restaurants).
In my implementation, I used the information of unique users in the
`Reviews` dataset along with the cosine similarity metric.

Note the reason for using cosine similarity is that we want to find if users
are similar or different, not how much they are similar or different (euclidean distance).

```py
# aggregate data for each unique user
data = reviews[['Reviewer Name', 'Birth Year', 'Marital Status', 'Has Children?',
                'Vegetarian?', 'Weight (lb)', 'Height (in)', 'Average Amount Spent',
                'Preferred Mode of Transport', 'Northwestern Student?']]

data_users = data.groupby(['Reviewer Name']).agg(**{
    'Birth Year': ('Birth Year', lambda x: pd.Series.mode(x)[0]),
    'Marital Status': ('Marital Status', lambda x: pd.Series.mode(x)[0]),
    'Has Children?': ('Has Children?', lambda x: pd.Series.mode(x)[0]),
    'Vegetarian?': ('Vegetarian?', lambda x: pd.Series.mode(x)[0]),
    'Weight (lb)': ('Weight (lb)', 'mean'),
    'Height (in)': ('Height (in)', 'mean'),
    'Average Amount Spent': ('Average Amount Spent', lambda x: pd.Series.mode(x)[0]),
    'Preferred Mode of Transport': ('Preferred Mode of Transport', lambda x: pd.Series.mode(x)[0]),
    'Northwestern Student?': ('Northwestern Student?', lambda x: pd.Series.mode(x)[0])
}).reset_index()
# convert categorical data into numerical
data_users = pd.get_dummies(
    data_users,
    columns=['Marital Status', 'Has Children?', 'Vegetarian?', 'Average Amount Spent',
             'Preferred Mode of Transport', 'Northwestern Student?'],
    drop_first=True, dtype=int
)
# remove user name
data_demographics = data_users.drop(columns=['Reviewer Name'])
# cosine distance of users
user_demo_cosine = pd.DataFrame(
    cosine_distances(data_demographics, data_demographics),
    columns=data_users['Reviewer Name'],
    index=data_users['Reviewer Name']
)
```

Finally, recommend items from someone that is most similar to the user.
In my implementation, I found the top n most similar users and find all restaurants
that they rated. Then, returned the restaurants that are the highest rated.

```py
def collab_filter_demographic(name, n_similar):
    most_similar_users = user_demo_cosine[user_demo_cosine.index != name][name].sort_values(ascending=True).index[0:n_similar]
    possible_recs = reviews[reviews['Reviewer Name'].isin(most_similar_users)].groupby(['Reviewer Name'])
    top_recs = possible_recs.apply(lambda x: x[x['Rating'] == x['Rating'].max()])
    return( top_recs[['Restaurant Name', 'Rating']].reset_index().drop(columns=['level_1']) )
```

You can locate my Collaborative Filtering Recommendation System with user information
in the `code` directory titled `collaborative1`.

### D2. Using information from the users related to the items

Recall that the purpose of finding similar users is to be able to find items to recommend
that a user would probably like. Previously, we are able to recommend items indirectly by
using information about the users to find similar users. However, we can also just
recommend items directly by using information about the users that is associated with the items.

In this case, instead of using information about the users, I will be using the
ratings from the users themselves in order to find similar users.
Let's illustrate an example. Suppose there are 5 users and 5 items.
A possible dataset can be:

|  | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 |
|-|-|-|-|-|-|
| User 1 | 5 | 1 | 4 |   |   |
| User 2 | 2 |   |   |   | 2 |
| User 3 | 4 | 3 | 1 | 4 | 3 |
| User 4 |   | 3 | 5 |   |   |
| User 5 | 5 |   | 2 |   | 1 |

Note that there exist a major issue with this type of information. It is rare that
a user has interated with most of the items. This issue is known as data sparsity.
A common solution is to use clustering. Rather than looking at individual users,
we can cluster together similar users and average their information.
Hopefully, this will lessen the problem with sparsity.
In our example, suppose be determine the clusters are (1,4); (2,5); (3).
Then, we find the average score per item within each cluster.

|  | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 |
|-|-|-|-|-|-|
| User 1,4  | 2.5   | 2 | 4.5   |   |       |
| User 2,5  | 3.5   |   | 2     |   | 1.5   |
| User 3    | 4     | 3 | 1     | 4 | 3     |

Nice! The data is less sparse. However, there is still missing data.
In this case, just perform some imputation to fill in the missing values.
In this example, I will just impute with the average score per item
across each cluster.

|  | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 |
|-|-|-|-|-|-|
| User 1,4  | 2.5   | 2     | 4.5   | 4 | 2.25  |
| User 2,5  | 3.5   | 2.5   | 2     | 4 | 1.5   |
| User 3    | 4     | 3     | 1     | 4 | 3     |

Note I have not covered how to perform the clustering.
There are several methods each with varying degrees of accuracy
like K-means, Gaussian Mixture Models, DBSCAN, etc. I will not be covering
clustering in-depth here, but I will show how to implement a simple K-means
clustering using information about the users in Python.

Ok, now lets' implement everything we covered above for our restaurant datasets.

(1) Find the restaurant ratings by user.

```py
# user scores of each restaurant
data_scores = reviews[['Reviewer Name', 'Restaurant Name', 'Rating']].groupby(['Reviewer Name', 'Restaurant Name']).agg(Rating = ('Rating', 'mean')).reset_index()
data_scores_table = data_scores.pivot(index='Restaurant Name', columns='Reviewer Name', values='Rating').reset_index()
# only consider restaurants that we have information on
restaurants_considered = list(restaurants['Restaurant Name'])
restaurants_considered.remove('Evanston Games & Cafe')
data_scores_table = data_scores_table[data_scores_table['Restaurant Name'].isin(restaurants_considered)]
```

(2) Perform K-means clustering to deal with sparseness.

Note that in my implementation, I chose 4 clusters. You can choose any number you want,
but there exist methods to find the optimal amount of clusters, but I won't be
covering it here.

```py
# first prepare the data for clustering
data_users_num = data_demographics[['Birth Year', 'Weight (lb)', 'Height (in)']]
data_users_cat = data_demographics.drop(columns=['Birth Year', 'Weight (lb)', 'Height (in)'])

scaler = StandardScaler()
data_users_scaled = scaler.fit_transform(data_users_num)
data_users_scaled = np.column_stack((data_users_scaled, data_users_cat))

# find which cluster than each user belong to
kmeans_labels = KMeans(n_clusters=4, n_init=10, max_iter=10).fit_predict(data_users_scaled)
```

(3) Using the clusters, find the average rating of each restaurant within each cluster.

```py
# merge user names with which cluster they belong to
data_users_clustered = np.column_stack((data_users['Reviewer Name'], kmeans_labels))
data_users_clustered = pd.DataFrame(data_users_clustered, columns=['Reviewer Name', 'cluster'])
data_scores_clustered = data_scores.merge(data_users_clustered, left_on='Reviewer Name', right_on='Reviewer Name')
# find average rating by restaurant for each cluster
avg_scores_clustered = data_scores_clustered.groupby(['cluster', 'Restaurant Name'])['Rating'].mean().unstack(level=0)
```

(4) Perform imputation for any remaining missingness.

In my implementation, I impute with the average score per restaurant across each cluster.
Note I melted the average clustered data for `value_vars` = $[0,1,2,3]$ because I used 4 clusters,
but if you use m clusters, the `value_vars` = $[0,1,2,..,m-1]$.

```py
# if ratings still missing, impute rating as average rating across clusters
avg_scores_clustered = avg_scores_clustered.apply(lambda row: row.fillna(row.mean()), axis=1).reset_index()
avg_scores_clustered = pd.melt(avg_scores_clustered, id_vars='Restaurant Name', value_vars=[0, 1, 2, 3]).reset_index()
# merge user name with their cluster data
data_user_avg_scores_clustered = data_users_clustered.merge(avg_scores_clustered, how='right', on='cluster')
data_user_avg_scores_clustered = data_user_avg_scores_clustered[['Reviewer Name', 'Restaurant Name', 'value']]
data_avg_scores_table = data_user_avg_scores_clustered.pivot(index='Restaurant Name', columns='Reviewer Name', values='value').reset_index()
data_avg_scores_table = data_avg_scores_table[data_avg_scores_table['Restaurant Name'].isin(restaurants_considered)]
# impute missingness in user scores of each restaurant
data_scores_table_complete = data_scores_table.fillna(data_avg_scores_table)
```

(5) Compute the cosine distance between each user and recommend items from
someone that is most similar to the user.


In my implementation, I found the top n most similar users and find all restaurants
that they rated. Then, returned the restaurants that are the highest rated.

```py
user_scores_cosine = pd.DataFrame(
    cosine_distances(
        data_scores_table_complete.drop(columns=['Restaurant Name']).T,
        data_scores_table_complete.drop(columns=['Restaurant Name']).T
    ),
    columns=data_scores_table_complete.columns[1:],
    index=data_scores_table_complete.columns[1:]
)

def collab_filter_scores(name, n_similar):
    most_similar_users = user_scores_cosine[user_scores_cosine.index != name][name].sort_values(ascending=True).index[0:n_similar]
    possible_recs = reviews[reviews['Reviewer Name'].isin(most_similar_users)].groupby(['Reviewer Name'])
    top_recs = possible_recs.apply(lambda x: x[x['Rating'] == x['Rating'].max()])
    return( top_recs[['Restaurant Name', 'Rating']].reset_index().drop(columns=['level_1']) )
```

You can locate my Collaborative Filtering Recommendation System with users' score ratings
in the `code` directory titled `collaborative2`.


