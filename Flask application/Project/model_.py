import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the dataset
df = pd.read_csv("C:/Users/bob_b/Downloads/archive/megaGymDataset.csv")

# Dropping duplicates
df.drop_duplicates(inplace=True, keep='first')

# Dropping NaN values
df.dropna(inplace=True)

# Dropping the RatingDesc column since it has all NaN or identical values
df.drop(columns=["RatingDesc"], inplace=True)

# Create a descriptive column based on the Rating
def impute_rating(row):
    if row['Rating'] == 0.0:
        return 'No Rating'
    elif row['Rating'] <= 4.0:
        return 'Below Average'
    elif row['Rating'] <= 7.0:
        return 'Average'
    else:
        return 'Above Average'

df['RatingDesc'] = df.apply(impute_rating, axis=1)

# Combine relevant columns for feature creation
df['features'] = df['BodyPart'].fillna('') + " " + df['Equipment'].fillna('') + " " + df['Level'].fillna('')

# Initialize TF-IDF Vectorizer and compute similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the TF-IDF model and similarity matrix for app usage
with open('tfidf_model.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('cosine_sim_matrix.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("TF-IDF model and similarity matrix saved.")

# Function to recommend exercises based on user input
def recommend_workout(muscle_group, num_exercises=5, exercise_level="Intermediate", equipment=None):
    # Filter exercises based on criteria
    filtered_df = df.copy()
    if muscle_group:
        filtered_df = filtered_df[filtered_df['BodyPart'].str.contains(muscle_group, case=False)]
    if exercise_level:
        filtered_df = filtered_df[filtered_df['Level'].str.contains(exercise_level, case=False)]
    if equipment:
        filtered_df = filtered_df[filtered_df['Equipment'].str.contains(equipment, case=False)]

    # Get top exercises
    workout_plan = []
    recommended_titles = set()

    for _, row in filtered_df.head(num_exercises * 2).iterrows():
        if row['Title'] not in recommended_titles:
            workout_plan.append(f"{row['Title']}: {get_sets_and_reps(exercise_level)}")
            recommended_titles.add(row['Title'])
        if len(workout_plan) >= num_exercises:
            break

    return [f"Day 1: {muscle_group.capitalize()}\n" + "\n".join(workout_plan)]

# Helper function to determine sets and reps
def get_sets_and_reps(exercise_level):
    level_mapping = {
        "Beginner": "3 sets of 10-12 reps",
        "Intermediate": "4 sets of 6-8 reps",
        "Advanced": "5 sets of 4-6 reps"
    }
    return level_mapping.get(exercise_level, "4 sets of 8-10 reps")

# Example: Generate top 5 recommended exercises
if __name__ == "__main__":
    muscle_group = "Chest"  # User-defined input
    equipment = "Dumbbell"  # User-defined input
    exercise_level = "Intermediate"  # User-defined input
    num_exercises = 5  # User-defined input

    workout_plan = recommend_workout(muscle_group, num_exercises, exercise_level, equipment)
    print(workout_plan)

# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load exercise dataset
# df = pd.read_csv("C:/Users/bob_b/Downloads/archive/megaGymDataset.csv")

# # Fill missing values with empty strings for concatenation
# df['BodyPart'] = df['BodyPart'].fillna('')
# df['Equipment'] = df['Equipment'].fillna('')
# df['Level'] = df['Level'].fillna('')
# df['Title'] = df['Title'].fillna('')  # Ensure 'Title' has no NaNs for recommendations

# # Combine relevant columns into a single text feature for TF-IDF
# df['features'] = df['BodyPart'] + " " + df['Equipment'] + " " + df['Level']

# # Initialize TF-IDF Vectorizer
# tfidf = TfidfVectorizer()
# tfidf_matrix = tfidf.fit_transform(df['features'])

# # Compute cosine similarity between exercises
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# # Save the model and vectorizer
# with open('tfidf_model.pkl', 'wb') as f:
#     pickle.dump(tfidf, f)
# with open('cosine_sim_matrix.pkl', 'wb') as f:
#     pickle.dump(cosine_sim, f)

# print("TF-IDF model and similarity matrix saved.")

# # Function to recommend exercises
# def recommend_workout(muscle_group, num_exercises=5, exercise_level="Intermediate", equipment=None):
#     # Filter exercises by muscle group, level, and equipment
#     filtered_df = df.copy()

#     # Apply muscle group filter
#     filtered_df = filtered_df[filtered_df['BodyPart'].str.contains(muscle_group, case=False)]
    
#     # Apply exercise level filter
#     filtered_df = filtered_df[filtered_df['Level'].str.contains(exercise_level, case=False)]
    
#     # Apply equipment filter if provided
#     if equipment:
#         filtered_df = filtered_df[filtered_df['Equipment'].str.contains(equipment, case=False)]

#     # Create a list to hold workout exercises
#     workout_plan = []
#     recommended_titles = set()  # To keep track of recommended exercises

#     # Loop through the filtered exercises and select the first `num_exercises`
#     for _, row in filtered_df.head(num_exercises * 2).iterrows():  # Iterate more to ensure we have a sufficient number of unique recommendations
#         exercise_title = row['Title']

#         # Skip exercises that have already been recommended (duplicates)
#         if exercise_title in recommended_titles:
#             continue
        
#         # Add exercise to the list and to the set of recommended exercises
#         workout_plan.append(f"{exercise_title}: {get_sets_and_reps(exercise_level)}")
#         recommended_titles.add(exercise_title)

#         # Stop when we have the desired number of recommendations
#         if len(workout_plan) >= num_exercises:
#             break

#     # Format the workout plan as Day 1, Day 2, etc., assuming a single workout day
#     return [f"Day 1: {muscle_group.capitalize()}\n" + "\n".join(workout_plan)]

# # Helper function to determine sets/reps based on exercise level
# def get_sets_and_reps(exercise_level):
#     if exercise_level == "Beginner":
#         return "3 sets of 10-12 reps"
#     elif exercise_level == "Intermediate":
#         return "4 sets of 6-8 reps"
#     elif exercise_level == "Advanced":
#         return "5 sets of 4-6 reps"
#     else:
#         return "4 sets of 8-10 reps"  # Default to intermediate

# # Example usage
# muscle_group = "Chest"  
# equipment = "Dumbbell"  
# workout_plan = recommend_workout(muscle_group, num_exercises=5, exercise_level="Intermediate", equipment=equipment)
# print(workout_plan)


# ######################################################################################################################
# # **Data Cleaning and Pre-processing:**
# # Importing the libraries and dataset

# import pandas as pd
# import numpy as np

# # Importing the dataset without Unnamed column
# df = pd.read_csv("C:/Users/bob_b/Downloads/archive/megaGymDataset.csv")
# df.head()
# # Checking the unique values of the column
# df.Type.unique()
# # Taking a look at the data: the last 5 rows
# df.tail()

# df[df.duplicated()]
# # Dropping the duplicates
# df.drop_duplicates(inplace=True, keep='first')
# # info about the dataset
# df.info()
# # Checking for unique values in the Rating and RatingDesc columns: 
# # We can impute the missing values in the RatingDesc column based on the Rating column
# print(df.Rating.unique())
# print(df.RatingDesc.unique())
# # Dropping all the NaN values
# df.dropna(inplace=True)
# # Dropping the RatingDesc column because it has all the values as either NaN or average
# df.drop(columns=["RatingDesc"], inplace=True)
# # Range of values in the Rating column
# df1 = df.Rating.sort_values()
# df1.unique()
# ## **Creating a new column and imputing ratingdesc based on Rating column:**
# # Creating a new column and imputing ratingdesc based on rating column
# def impute_rating(row):
#     if row['Rating'] == 0.0:
#         return 'No Rating'
#     elif row['Rating'] <= 4.0:
#         return 'Below Average'
#     elif row['Rating'] <= 7.0:
#         return 'Average'
#     else: 
#         return 'Above Average'

# df['RatingDesc'] = df.apply(lambda row: impute_rating(row), axis=1)

# df.head()
# df.isnull().sum()
# ### Now that our data is clean, we can move forward in life and do other important things like doing EDA:
# # **Main Goals:**
# ## 1. **Top 5 Rated Exercises for each body part:**
# # to show all rows in jupyter output
# pd.set_option('display.max_rows', None)
# # Group by 'BodyPart' and then apply the top 5 highest rated exercises for each group
# df1 = df.groupby('BodyPart').apply(lambda x: x.nlargest(5, 'Rating')).reset_index(drop=True)

# # Sort by 'BodyPart' and 'Rating'
# df1 = df1.sort_values(by=['BodyPart', 'Rating'], ascending=[True, False])
# # Show only the columns 'BodyPart', 'Title', and 'Rating'
# df1[['BodyPart','Title','Rating']]

# ## 2. **Top 5 rated exercises for each Type / Category of Workouts:**
# # Group by 'Type' and then apply the top 5 highest rated exercises for each group
# df2 = df.groupby('Type').apply(lambda x: x.nlargest(5, 'Rating')).reset_index(drop=True)

# # Sort by 'BodyPart' and 'Rating'
# df2 = df2.sort_values(by=['Type', 'Rating'], ascending=[True, False])
# # Show only the columns 'BodyPart', 'Title', and 'Rating'
# df2[['Type','BodyPart','Title','Rating']]

# ## 3. **Top 5 rated exercises for each BodyPart and each Level of exercises:**
# # Group by 'Level' and 'BodyPart' and then apply the top 5 highest rated exercises for each group
# df2 = df.groupby(['Level', 'BodyPart']).apply(lambda x: x.nlargest(5, 'Rating')).reset_index(drop=True)

# # Sort by 'BodyPart' and 'Rating'
# df2 = df2.sort_values(by=['Level', 'BodyPart', 'Rating'], ascending=[True, True, False])
# df2[['Level','BodyPart','Title','Rating']]

# ## 5. **To ask the user about Type, BodyPart, Equipment he / she has, and the Level of Fitness he is at and recommend top exercises based on demand:**
# # Code to print unique values for each of these columns so that we can add those exact strings to get desired output exercises
# columns = ['Type', 'BodyPart', 'Equipment', 'Level']
# for col in columns:
#     unique_values = df[col].unique()
#     print(f"Unique values in '{col}':\n {unique_values}")

# # You can replace the values based on above given output to get filtered exercises
# filtered_df = df.loc[
#     (df['Type'] == 'Strength') & 
#     (df['BodyPart'] == 'Biceps') & 
#     (df['Equipment'] == 'Dumbbell') & 
#     (df['Level'] == 'Beginner')
# ]
# filtered_df