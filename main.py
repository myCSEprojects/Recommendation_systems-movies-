import streamlit as st
import pandas as pd
import numpy as np
from initial_recommendation import get_recommendations
from page_manager import get, set, store, get_user, store_ratings, get_ratings, store_topmovies, get_topmovies
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from final_movie_recommendations import matrix_factorization


latent_k = 3

def user_input_pg():

    st.title("User Details")

    with st.form("my_form"):
        name = st.text_input("Enter your name").strip()
        age = st.text_input("Enter your age").strip()
        gender = st.selectbox("Select your gender", ["Male", "Female"])
        submitted_user = st.form_submit_button("Submit")

    if (submitted_user):
        if (name== ""):
            st.error("Name cannot be empty")
            st.stop()

        try:
            age = int(age)
              
        except:
            st.error("Invalid age format, age must be an integer")
            st.stop()


        gender = 'M' if gender=='Male' else 'F'

        set(2)
        store(age,gender)
        st.write("Thank you for providing your details, {}!.".format(name))
    
  
    button_style = "text-align:center;"

    if st.button("Show Movies to Rate"):
        pass


pg = get()
if (pg==1):
    user_input_pg()
    
elif pg == 2:
    st.title("Rate your Movies")
    names,indices = get_recommendations(get_user()[0], get_user()[1])
    ratings = []

    with st.form("rating_form"):
        for i, item in enumerate(names):
            st.write(f"**{item}**")
            rating = st.selectbox(f"Select your rating", ["0", "1", "2", "3", "4", "5"], key=f"rating_{i}")
            ratings.append(int(rating))
        
        
        submitted = st.form_submit_button("Submit")

    count0 = ratings.count(0)

    if submitted: 
        if count0>25:
            st.error("Please rate at least 10 movies")
            st.stop()
        
        ratings_ = []
        indices_ = []
        for i in range(len(indices)):
            if (ratings[i]!=0):
                ratings_.append(ratings[i])
                indices_.append(indices[i])

        store_ratings(ratings_,indices_)
        st.write("Thank you for providing your ratings!")
        set(3)

    

    if st.button("Show Recommendations"):

        pass

elif (pg==3): 
    df = pd.read_csv("latent_space.csv")
    df.columns = df.columns.astype(int)
    cols = list(df.columns.values)

    movies_latent_flag = np.array(df.values)

    movies_latent = torch.tensor(df.values, dtype=torch.float)

    rat,ind = get_ratings()
    print(rat)
    dic_ratings = dict(zip(ind,rat))

    available_ratings = []
    for i in cols:
        try:
            a = dic_ratings[i]
            available_ratings.append(a)
        except:
            available_ratings.append(np.nan)


    available_ratings = torch.tensor(available_ratings)

    user_latent = torch.randn(1,3, dtype = torch.float, requires_grad=True)
    optimiser = optim.Adam([user_latent], lr=0.01)


    for i in range(1000):
        pred_ratings = torch.mm(user_latent,movies_latent)
        squashed_rating = torch.sigmoid(pred_ratings)
        scaled_rating = (squashed_rating * 4.0) + 1.0

        diff_vector = scaled_rating - available_ratings

        
        mask = ~torch.isnan(diff_vector)
        diff_vector = diff_vector[mask]
        loss = torch.norm(diff_vector)

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        if (i%100==0):
            print(loss.item())

    user_latent = user_latent.detach().numpy()

    learnt_predictions = np.dot(user_latent, movies_latent_flag)
    squashed_rating = 1/(1+np.exp(-learnt_predictions))
    scaled_rating = (squashed_rating * 4.0) + 1.0

    scaled_rating = scaled_rating.reshape((-1,1))
    
    final_df = pd.DataFrame({1:list(scaled_rating), 2: list(cols)})

    final_df = final_df.sort_values(by=1, ascending=False)

    top_5 = final_df.head(5)[2].values.tolist()
    top_5_ratings = final_df.head(5)[1].values.tolist()
    top_5_ratings_ = []
    for i in top_5_ratings:
        for j in i:
            top_5_ratings_.append(j)

    #Loading data for matching movie names
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

    movies = movies.set_index('movie_id', drop=True)

    # final_input_movies = movies[movies['movie_id'].isin(final_recommendation.astype(str))]['title'].values.tolist()
    USER_MOVIES = []
    USER_MOVIES_ = []

    for i in range(len(top_5)):
        USER_MOVIES.append([movies.loc[str(top_5[i])]['title'], top_5_ratings_[i]])
        USER_MOVIES_.append(movies.loc[str(top_5[i])]['title'])
    
    USER_MOVIES = pd.DataFrame(USER_MOVIES, columns=['Movie', 'Rating'], index = [1,2,3,4,5])

    st.title("Your Recommendations")
    st.write("Here are the top 5 movies we recommend you to watch")
    # for i,j in USER_MOVIES:
    #     st.write(f"***{i}***  :  {j}")

    st.table(USER_MOVIES)


    store_topmovies(top_5, USER_MOVIES_)

    set(4)
    # Reevaluate the dataset by taking the feedback ratings from the user

    if st.button("Give Feedback"):
        pass

elif (pg==4):
    st.title("Feedback")
    st.write("Please rate all the movies that we recommended you")

    top_5, USER_MOVIES = get_topmovies()
    ratings = []
    with st.form("feedback_form"):
        for i, item in enumerate(USER_MOVIES):
            st.write(f"**{item}**")
            rating = st.selectbox(f"Select your rating", [ "1", "2", "3", "4", "5"], key=f"rating_{i}")
            ratings.append(int(rating))
        
    
        submitted = st.form_submit_button("Submit")


    if submitted:
        st.write("Thank you for providing your feedback!")
        all_ratings = get_ratings()[0] + ratings
        all_indices = get_ratings()[1] + top_5

        df = pd.read_csv("latent_space.csv")
        df.columns = df.columns.astype(int)
        cols = list(df.columns.values)

        dic_ratings = dict(zip(all_indices,all_ratings))

        available_ratings = []
        for i in cols:
            try:
                a = dic_ratings[i]
                available_ratings.append(a)
            except:
                available_ratings.append(np.nan)
        
        matrix_factorization(available_ratings)
        set(3)
        if st.button("Return to Main Page",key = 2):
            set(1)

        if st.button("Show Recommendations",key = 3):
            set(3)
        

        











        