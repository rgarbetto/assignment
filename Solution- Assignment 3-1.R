
library(quanteda)
library(tidyverse)

# 1) Read in the airbnb data to R and randomly select 1000 rows. Do all the preprocessing on the variable 'comments'
# except stemming and perform a TFIDF weighting on the dataset, and then transpose it.
# Check first 5 rows and columns.

set.seed(124563)
airbnb<-read_csv("sample_airbnb.csv")
airbnb_tfidf <- tokens(airbnb$comments, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, split_hyphens = TRUE)%>%
                tokens_remove(stopwords(source = "smart")) %>%
                tokens_tolower() %>%
                dfm() %>%
                dfm_tfidf(scheme_tf = "prop", scheme_df = "inverse", base = 10)%>%
                t
airbnb_tfidf[1:5,1:5] # check first 5 rows and columns

# 2) Load the relevant libraries for performing Latent Semantic Analysis. Use the 
# appropriate R function to implement the LSA model and automatically generate 
# the truncated U, Sigma and V matrices. Examine the list generated for the dimensions
# of the matrices. What would be dimension of the product of truncated U, Sigma
# and V matrices. Should there be a relation between the dimension of dataset generated
# in step 1 and the product of truncated U, Sigma and V matrices?

library(lsa)
library(LSAfun)
airbnb_LSAspace <- lsa(airbnb_tfidf, dims=dimcalc_share())

# The dimensions of the original dataset and product of truncated matrices should be same
dim(airbnb_tfidf)
dim(airbnb_LSAspace$tk)
dim(airbnb_LSAspace$dk)
length(airbnb_LSAspace$sk)# Why didn't we do dim here?

#3) Implement the LSA model to generate truncated matrices with dimension=4, and answer the same questions as in question 2.
airbnb_LSAspace1 <- lsa(airbnb_tfidf, dims=4)

#4) When we multiply the truncated matrices, how can we interpret the output?

# This text segment is best described as having so much of abstract concept one and
# so much of abstract concept two (if we use k=2), and this word has so much of concept one and so
# much of concept two, and combining those two pieces of information (by vector
# arithmetic), my best guess is that word X actually appeared 0.6 times in context Y.

#5) Multiply the truncated $??,U$ matrices from question 2 with appropriate tranposes
# to compute cosine similarity between terms 'apartment' and 'comfortable'. And also
# between 'apartment' and 'issues')
tk2 = t(airbnb_LSAspace$sk * t(airbnb_LSAspace$tk))
myCo <- costring('apartment','comfortable', tvectors= tk2, breakdown=TRUE)
myCo
myCo1 <- costring('apartment','issues', tvectors= tk2, breakdown=TRUE)
myCo1

#6) Compute cosine similarity between all the terms, then look at the first 7 rows and columns

myTerms2 <- rownames(tk2)
myCosineSpace2 <- multicos(myTerms2, tvectors=tk2, breakdown=TRUE)
#breakdown=TRUE forces data into lower case
myCosineSpace2[1:7,1:7]


# 7) Similar to the term matrices, multiply the $??×V^T$ matrices for document 
# comparisons, and compute all cosine similarities, then look at the first 7 rows and columns


dk2 = t(airbnb_LSAspace$sk * t(airbnb_LSAspace$dk))
dk2[1:10,1:3]
myTerms2 <- rownames(dk2)
myCosineSpace2 <- multicos(myTerms2, tvectors=dk2)
myCosineSpace2[1:7,1:7]

# 8) Find the 5 nearest neighbors of 'text10', 'problems', then plot 5 neighbors of 'features'

neighbors("text10", n=5, tvectors = dk2, breakdown = F)
neighbors("food", n=5, tvectors = tk2, breakdown = T)
plot_neighbors("awesome", n=5, tvectors = tk2)



#10) Another function that returns terms that are close to a given term is associate. While neighbors returns
# the nearest n terms, associate returns whichever terms are within a particular cutoff.
# Use 'associate' with cutoff 0.35 for word 'restaurant'

associate(tk2, "restaurant", measure="cosine", threshold=0.35)

# 11) A convenient method of eyeballing what the data may indicate is to run a correlation on the terms matrix or on the documents matrix. Because of
# the structure of the tk and dk matrices, one needs to transpose them first. The function t does that. The
# correlation function is called cor. By default, cor runs a Pearson correlation. In the example below, we
# force it to run a Spearman and then a Kendall correlation just to show that it one can do it. The function
# cor can also be set to treat missing values as a listwise ("complete.obs") or as a pairwise deletion
# ("pairwise.complete.obs"). These correlations provide insight on how the data might be interrelated. In
# this case, there is a clear grouping into two sets of terms and two sets of documents.
# I am showing two commands below, use the help file to understand them better.
trans_tk <- t(as.matrix(tk2))
trans_dk <- t(as.matrix(dk2))
cor_terms<-cor(trans_tk, use="complete.obs", method="spearman")
#cor_docs<-cor(trans_dk, use="pairwise.complete.obs", method="kendall")
cor_terms[1:7,1:7]
