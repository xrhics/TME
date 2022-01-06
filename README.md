# TME
Tree-guided Multi-task Embedding Learning for Semantic Annotation of Venues Using Check-ins.
TME.py is a file that learns presentation of venues and annotates semantics of venues.

## Data
* The input file "trajectoryTraining.txt" is from Foursquare, including check-ins of 18 months collected from Tokyo and New York. Each line of this file is a check-in sequence of a user. The format is: [venueid venueid venueid ...].

* The input file "category0.csv" contains the venue categories used by Foursquare, which constitute a five-layer hierarchical structure, as the category hierarchy. The format is: [categoryid,categoryname,parent]

## For Example: 
"trajectoryTraining.txt": 

4f6c5507121d7483f0ac8610 4b6cf0daf964a5202f5f2ce3 4aed0b8cf964a520ebcc21e3 4b4020f5f964a520f7b525e3 4b6cf0daf964a5202f5f2ce3 4b6cf0daf964a5202f5f2ce3 4b6cf0daf964a5202f5f2ce3 4b683963f964a520c66c2be3 4ec67ae75c5ce271bc51bfae  
4b7ecbfdf964a520330030e3 4bd44b84a8b3a5937e7f6b5f 4bf2c3a0af659c748809d847 4b058806f964a52067ad22e3...

"category0.csv":

4d4b7105d754a06379d81259,Travel & Transport,root  
4d4b7105d754a06378d81259,Shop & Service,root  
4d4b7105d754a06372d81259,College & University,root  
4d4b7105d754a06373d81259,Event,root  
4d4b7105d754a06374d81259,Food,root  
4d4b7104d754a06370d81259,Arts & Entertainment,root  
4d4b7105d754a06377d81259,Outdoors & Recreation,root  
4d4b7105d754a06376d81259,Nightlife Spot,root  
52f2ab2ebcbc57f1066b8b57,Recruiting Agency,4d4b7105d754a06375d81259$root  
4c38df4de52ce0d596b336e1,Parking,4d4b7105d754a06375d81259$root  
4bf58dd8d48988d172941735,Post Office,4d4b7105d754a06375d81259$root...


