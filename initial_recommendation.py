import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def get_recommendations(user_age, user_gender):
    # Read in the data as a text file
    with open("ml-1m/ratings.dat", "r") as f:
        data = f.readlines()

    # Split each line into separate fields
    data = [line.strip().split("::") for line in data]

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])

    # Remove the timestamp column
    df = df.drop("timestamp", axis=1)

    # Convert the rating column to integers
    df["rating"] = df["rating"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["user_id"] = df["user_id"].astype(int)


    data=df

    data["user_id"].unique().shape
    data["item_id"].unique().shape

    ratings = data.pivot(index='user_id', columns=('item_id'), values='rating')

    data = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",
                    names=["user_id", "gender", "age", "occupation_id", "time_stamp"])

    # select columns of interest
    user_data = data[["user_id", "gender", "age"]]

    for i in range(len(user_data['gender'])):
        if user_data['gender'][i]=='M':
            user_data['gender'][i] = 1
        else:
            user_data['gender'][i] = 0

    user_details = [user_age,user_gender,user_data]

    def nearest_neighbours(age, gender,df):
        if (gender == 'M') : gender = 1
        else: gender = 0

        age_ = np.array(df['age'])
        gender_ = np.array(df['gender'])

        distances = (np.square(age - age_) + np.square(gender - gender_))

        distances =  distances**0.5 

        sorted_df = pd.DataFrame({'user_id':[i+1 for i in range(df.shape[0])], 'distances' : distances})
        sorted_df = sorted_df.sort_values('distances', ascending=True)

        sorted_df_ = sorted_df[sorted_df['distances']<=5]

        check_size = sorted_df_.shape[0]
        if (check_size<30):
            sorted_df_ = sorted_df.iloc[:31]
    
        return np.array(sorted_df_['user_id'])


    def generate_movies(user_details, n_neighbours, ratings):
        neighbours = nearest_neighbours(user_details[0], user_details[1], user_details[2])
        nearest_random_neighbours = np.random.choice(neighbours, n_neighbours, replace=False)

        movies = []
        for user in nearest_random_neighbours:  
            user_movies = []
            #li = (ratings.iloc[user].values.tolist())
            for col in ratings:
                if ratings[col][user] is not np.nan:
                    user_movies.append(col)
            
            movies+= list(np.random.choice(np.array(user_movies), 3, replace=False))
        
        return np.unique(np.array(movies))


    movies_to_recommend = 30
    final_recommendation = np.random.choice((generate_movies(user_details = user_details, n_neighbours=30, ratings=ratings)), movies_to_recommend, replace=False)


    # Define the path to the movies.dat file
    movies_file = 'ml-1m/movies.dat'

    # Define the columns in the movies.dat file
    columns = ['movie_id', 'title', 'genres']

    # Read the movies.dat file into a list of strings
    with open(movies_file, 'r', encoding='ISO-8859-1') as f:
        movies_list = f.readlines()

    # Split each line in the list into a list of values
    movies_list = [line.strip().split('::') for line in movies_list]

    # Create a pandas DataFrame from the list of values
    movies = pd.DataFrame(movies_list, columns=columns)

    final_input_movies = movies[movies['movie_id'].isin(final_recommendation.astype(str))]['title'].values.tolist()

    return final_input_movies, list(final_recommendation)
