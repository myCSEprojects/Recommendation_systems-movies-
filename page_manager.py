page = 1
age = -1
gender = ''
ratings = []
indices = []
topmovies = []
topindices = []

def get():
    return page

def set(value):
    global page
    page = value

def store(age_,gender_):
    global age,gender
    age = age_ 
    gender = gender_ 

def get_user():
    return age,gender

def store_ratings(ratings_, indices_):
    global ratings, indices
    ratings = ratings_
    indices = indices_

def get_ratings():
    return ratings, indices

def store_topmovies(topindices_,topmovies_):
    global topmovies,topindices
    topmovies = topmovies_
    topindices = topindices_

def get_topmovies():
    return topindices,topmovies
