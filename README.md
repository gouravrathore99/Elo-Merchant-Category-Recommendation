# Elo Merchant Category Recommendation Case Study

Elo is one of the largest domestic debit and credit card brands of Brazil. In February 2019 it organized 'Elo Merchant Category Recommendation' competition on Kaggle. Elo wanted to provide discount and offers to its cardholders, for which it partnered with merchants. But Elo didn't want these discounts and offers to be random but specifically personalized for their individual cardholder based on their preference, current locality and other needs. Elo had already built machine learning models which can understand preferences and other aspects of a cardholder's life-cycle from food to shopping, but they had one major limitation. The models weren't designed specifically for individual cardholders but as a group.

Elo wants to reward its cardholders with promotions and discounts at selected merchant outlets which are relevant to those cardholders. These rewards depend on the loyalty of the cardholders towards Elo brand. In simple term, the cardholder who is frequent user of Elo's debit and/or credit cards will be rewarded with more offers at cardholder's frequent or favorite merchants. Elo has provided Loyalty score as a metric to measure the loyalty of cardholders toward Elo. The Loyalty score is a numerical score of each cardholder depending upon their preferences and transaction behavior.
In this competition, we had to develop algorithms which learn from each individual cardholder's previous preferences and activities, and identify and serve the most relevant opportunities, by predicting cardholder's loyalty score. The cardholder's loyalty score will help ELO to provide fully personalized promotions and discounts with merchants to individual cardholder. This will not only enrich the cardholder experience but also act as good marketing strategy for ELO to keeps the existing cardholder for repeat business and brings forth many new customers. It also reduces ELO's futile effort for unwanted campaign and let it focus on the area's where it is required.

The objective is to train a machine learning model which predict the Loyalty score for each cardholder. Since the loyalty score is a continuous/real variable, this is a regression problem. The Loyalty score will depend on cardholder's preferences and past transactions behavior. So, these preferences and behavior will have to provided to the model in numerical form.
The prediction of Loyalty score need not be done instantly as Elo will most likely store Loyalty score of cardholders beforehand and use it whenever the cardholders have to be provided with promotions and discount. Though the predictions will have to be updated with changes in cardholder's behavior, choices, increase or decrease in transaction and other activities.
