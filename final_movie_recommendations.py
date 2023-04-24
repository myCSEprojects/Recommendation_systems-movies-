# Generic Matrix Factorization (without missing values)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


# Read in the data as a text file
with open("ml-1m/ratings.dat", "r") as f:
    data = f.readlines()

# Split each line into separate fields
data = [line.strip().split("::") for line in data]
df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])
df = df.drop("timestamp", axis=1)
df["rating"] = df["rating"].astype(int)
df["item_id"] = df["item_id"].astype(int)
df["user_id"] = df["user_id"].astype(int)

data=df

A = data.pivot(index='user_id', columns='item_id', values='rating')

def matrix_factorization(new_user_data=None):

    global A
    if (new_user_data is not None):
        A.loc[len(A)+1] = new_user_data

    print(A.shape)
    print(1111111111111)

    cols = A.columns.values
    A = torch.tensor(A.values)
    mask = ~torch.isnan(A)

    #storing indices of each rating in the form of (user, movie)
    users, movies = torch.where(mask)

    ratings = A[mask]

    # Store in PyTorch tensors
    users = users.to(torch.int64)
    movies = movies.to(torch.int64)
    ratings = ratings.to(torch.float32)

    # Now use matrix factorization to predict the ratings

    # Create a class for the model

    class MatrixFactorization(nn.Module):
        def __init__(self, n_users, n_movies, n_factors=20):
            super().__init__()
            self.user_factors = nn.Embedding(n_users, n_factors)
            self.movie_factors = nn.Embedding(n_movies, n_factors)

        def forward(self, user, movie):

            valid_user = self.user_factors(user)
            valid_movie = self.movie_factors(movie)

            squashed_rating = torch.sigmoid((valid_user * valid_movie).sum(1))
            scaled_rating = (squashed_rating * 4.0) + 1.0

            return scaled_rating
        

        def return_result(self):
            return self.user_factors.weight.data, self.movie_factors.weight.data
        

    # Fit the Matrix Factorization model

    def factorize(k, n_users, n_movies):
        model = MatrixFactorization(n_users, n_movies, k)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for i in range(1000):
            # Compute the loss1``
            pred = model.forward(users, movies)
            loss = F.mse_loss(pred, ratings)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backpropagate
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Print the loss
            # if i % 100 == 0:
            #     print(loss.item())
        return model, loss.item()


    n_users, n_movies = A.shape
    k=3

    model, loss = factorize(k,n_users,n_movies)

    W,H = model.return_result()

    w,h = np.array(W), np.array(H)

    df = pd.DataFrame(h.T,columns=cols)

    df_ = pd.DataFrame(W)

    A = pd.DataFrame(A.numpy(),columns=cols)

    df.to_csv('latent_space.csv', index=False)
    df_.to_csv('user_space.csv', index=False)


if __name__ == "__main__":
    matrix_factorization()