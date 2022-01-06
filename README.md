# TME
Tree-guided Multi-task Embedding Learning for Semantic Annotation of Venues Using Check-ins.

## Data
* The input file "trajectoryTraining𝑥.txt" is from Foursquare, including check-ins of 18 months collected from Tokyo and New York. Each line of this file is a check-in sequence of a user. Note: we randomly mark off 𝑥% (x% = 10%, 20%, 30%) of all the venues and replace their categories with the “NULL” tag.

* The input file "category0.csv" contains the venue categories used by Foursquare, which constitute a five-layer hierarchical structure, as the category hierarchy. The format is: [categoryid,categoryname,parent]

## For Example: 
"trajectoryTraining10.txt": 

4b2aef90f964a52067b224e3#Bookstore,4bc39536920eb71316ab1d2c#NULL,4bc39536920eb71316ab1d2c#NULL,4b6119e9f964a520b2092ae3#NULL,4b8c4178f964a520dac632e3#Airport,4baf0f0bf964a52021e93be3#Plaza,4b2ba998f964a520d8b824e3#Multiplex,4ba9a681f964a520e6323ae3#Train Station,4ba9a681f964a520e6323ae3#Train Station  
4b89f269f964a520f25732e3#Airport,4b89f269f964a520f25732e3#Airport,4b2ba998f964a520d8b824e3#Multiplex,502354bfe4b053b46dd97c82#Shopping Mall,4efa453a9a52c908439fc8da#Train ...

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


