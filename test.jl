import Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("StatsPlots")
Pkg.add("StatsBase")
Pkg.add("StatsModels")
Pkg.add("GLM")
Pkg.add("Distributions")
Pkg.add("HypothesisTests")
Pkg.add("ScikitLearn")
Pkg.add("TextAnalysis")
Pkg.add("Word2Vec")
Pkg.add("Random")
Pkg.add("Statistics")

using TextAnalysis
using Word2Vec
using DataFrames
using CSV
using ScikitLearn
using Statistics
using Random

# Assuming df is your DataFrame and it has columns 'advert', 'price', 'rooms', 'footage'
df = CSV.read("your_data.csv")

# Preprocess and tokenize the text
stop_words = stopwords(ENGLISH)
df[!, :tokenized_advert] = map(x -> [word for word in tokenize(lowercase(x)) if isalpha(word) && !(word in stop_words)], df[:, :advert])

# Train Word2Vec model
word2vec_model = word2vec(df[:, :tokenized_advert], 100, 5, 2, 0)

# Create document vectors by averaging word vectors for each advert
df[!, :doc_vector] = map(x -> mean([get_vector(word2vec_model, word) for word in x if has_word(word2vec_model, word)], dims=1), df[:, :tokenized_advert])

# Handle missing values (if any word is not in the Word2Vec vocabulary)
dropmissing!(df)

# Split data into train and test sets
train_ind, test_ind = partition(1:nrow(df), 0.8)
train_df, test_df = df[train_ind, :], df[test_ind, :]

# Create train and test data for the model
@sk_import ensemble:RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

X_train = hcat(train_df[!, :rooms], train_df[!, :footage], convert(Matrix, train_df[!, :doc_vector]))
y_train = train_df[!, :price]

X_test = hcat(test_df[!, :rooms], test_df[!, :footage], convert(Matrix, test_df[!, :doc_vector]))
y_test = test_df[!, :price]

# Train the model
fit!(model, X_train, y_train)

# Predict on test data
y_pred = predict(model, X_test)

# Evaluate the model
mae = mean(abs.(y_test - y_pred))
mse = mean((y_test - y_pred) .^ 2)

println("Mean Absolute Error: ", mae, ", Mean Squared Error: ", mse)
