# # # import numpy as np
# # # from flask import Flask, request, render_template
# # # import pickle

# # # # Initialize Flask
# # # app = Flask(__name__, template_folder='template')

# # # # Load the trained model
# # # model = pickle.load(open('model.pkl', 'rb'))

# # # @app.route('/')
# # # def home():
# # #     return render_template('index.html')

# # # @app.route('/', methods=['POST'])
# # # def predict():
# # #     # Get form values and convert to integers
# # #     int_features = [int(x) for x in request.form.values()]
# # #     final_features = [np.array(int_features)]
# # #     prediction = model.predict(final_features)

# # #     # Round the output to 2 decimal places
# # #     output = round(prediction[0], 2)

# # #     # Check for negative prediction
# # #     if output < 0:
# # #         return render_template('index.html', prediction_text="Predicted Price is negative, values entered not reasonable")
# # #     else:
# # #         return render_template('index.html', prediction_text='Predicted Price of the house is: ${}'.format(output))

# # # if __name__ == "__main__":
# # #     app.run(debug=True)


# # import numpy as np
# # import pandas as pd
# # import pickle
# # from flask import Flask, request, render_template

# # # Initialize Flask
# # app = Flask(__name__, template_folder='template')

# # # Load the TF-IDF model and cosine similarity matrix
# # with open('tfidf_model.pkl', 'rb') as f:
# #     tfidf = pickle.load(f)
# # with open('cosine_sim_matrix.pkl', 'rb') as f:
# #     cosine_sim = pickle.load(f)

# # # Load the exercise dataset
# # df = pd.read_csv("C:/Users/bob_b/Downloads/archive/megaGymDataset.csv")

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/', methods=['POST'])
# # def recommend():
# #     # Get the exercise name from the form
# #     exercise_name = request.form.get("exercise_name")
# #     num_recommendations = int(request.form.get("num_recommendations", 3))

# #     # Check if the exercise exists in the dataset and get recommendations
# #     try:
# #         index = df[df['Title'] == exercise_name].index[0]
# #         sim_scores = list(enumerate(cosine_sim[index]))
# #         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# #         sim_scores = sim_scores[1:num_recommendations + 1]  # Exclude the first result (same exercise)
# #         recommendations = [df['Title'].iloc[i[0]] for i in sim_scores]
# #     except IndexError:
# #         return render_template('index.html', recommendation_text=f"Exercise '{exercise_name}' not found.")

# #     # Return the recommendations
# #     return render_template('index.html', recommendation_text='Recommended Exercises: ' + ', '.join(recommendations))

# # if __name__ == "__main__":
# #     app.run(debug=True)

# from flask import Flask, request, render_template
# import pickle
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Initialize Flask app
# app = Flask(__name__, template_folder='Template')

# # Load the exercise dataset and model
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

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def recommend():
#     # Get form values
#     muscle_group = request.form['muscle_group']
#     num_exercises = int(request.form['num_exercises'])
#     exercise_level = request.form['exercise_level']
#     equipment = request.form.get('equipment', None)

#     # Get recommendations
#     recommendations = recommend_workout(muscle_group, num_exercises, exercise_level, equipment)

#     return render_template('index.html', recommendation_text='Recommended Exercises: ' + ', '.join(recommendations))

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

#     # Loop through the filtered exercises and select the first `num_exercises`
#     for _, row in filtered_df.head(num_exercises).iterrows():
#         exercise_title = row['Title']
#         sets_and_reps = get_sets_and_reps(exercise_level)  # Get sets/reps based on exercise level
#         workout_plan.append(f"{exercise_title}: {sets_and_reps}")

#     # Format the workout plan as Day 1, Day 2, etc., assuming single workout day
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

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__, template_folder='Template')

# Load and clean the exercise dataset
df = pd.read_csv("C:/Users/bob_b/Downloads/archive/megaGymDataset.csv")

# Preprocess the dataset
df.drop_duplicates(inplace=True, keep='first')  # Drop duplicates
df.dropna(inplace=True)  # Drop rows with NaN values
df['BodyPart'] = df['BodyPart'].fillna('')
df['Equipment'] = df['Equipment'].fillna('')
df['Level'] = df['Level'].fillna('')
df['Title'] = df['Title'].fillna('')

# Create a descriptive column for ratings
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

# Combine relevant columns into a single text feature for TF-IDF
df['features'] = df['BodyPart'] + " " + df['Equipment'] + " " + df['Level']

# Initialize TF-IDF Vectorizer and compute similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the TF-IDF model and cosine similarity matrix
with open('tfidf_model.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('cosine_sim_matrix.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def recommend():
    # Get form values
    muscle_group = request.form.get('muscle_group', '')
    num_exercises = int(request.form.get('num_exercises', 5))
    exercise_level = request.form.get('exercise_level', 'Intermediate')
    equipment = request.form.get('equipment', None)

    # Generate recommendations
    recommendations = recommend_workout(muscle_group, num_exercises, exercise_level, equipment)

    # Display recommendations in the template
    return render_template('index.html', recommendation_text='Recommended Exercises:\n' + '\n'.join(recommendations))

# Function to recommend exercises
def recommend_workout(muscle_group, num_exercises=5, exercise_level="Intermediate", equipment=None):
    # Filter exercises by criteria
    filtered_df = df.copy()
    if muscle_group:
        filtered_df = filtered_df[filtered_df['BodyPart'].str.contains(muscle_group, case=False)]
    if exercise_level:
        filtered_df = filtered_df[filtered_df['Level'].str.contains(exercise_level, case=False)]
    if equipment:
        filtered_df = filtered_df[filtered_df['Equipment'].str.contains(equipment, case=False)]

    # Select the top exercises
    workout_plan = []
    for _, row in filtered_df.head(num_exercises).iterrows():
        exercise_title = row['Title']
        sets_and_reps = get_sets_and_reps(exercise_level)
        workout_plan.append(f"{exercise_title}: {sets_and_reps}")

    return workout_plan

# Helper function to determine sets/reps
def get_sets_and_reps(exercise_level):
    level_mapping = {
        "Beginner": "3 sets of 10-12 reps",
        "Intermediate": "4 sets of 6-8 reps",
        "Advanced": "4 sets of 4-6 reps"
    }
    return level_mapping.get(exercise_level, "4 sets of 8-10 reps")

if __name__ == "__main__":
    app.run(debug=True)
