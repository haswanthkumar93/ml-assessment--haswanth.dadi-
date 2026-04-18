# Business Case Analysis: Promotion Effectiveness at a Fashion Retail Chain.

## B1(a) Problem Formulation.

The goal of this machine learning project is forecasting items sold per store during specific monthly promotion campaigns.
-Target Variable.
-items_sold.

Represents total products sold at a store over the promotion timeframe.
-Potential Input Features
-promotion_type (e.g., Flat Discount, BOGO, Free Gift).
-store_location_type (urban, semi-urban, rural).
-store_size
-monthly_footfall
-competition_density
-customer_demographics
-month/season
-historical promotion performance
These factors shape promotion success across stores.

ML Problem Type:
Supervised regression.
Supervised: Uses labeled historical data with inputs (store/promotion details) and outputs (items_sold).
Regression: Predicts continuous numerical values, not categories.

## B1(b) Why Items Sold is a Better Target Variable.

Items sold trumps total sales revenue as a target because revenue fluctuates wildly with prices, promotions, and item varieties. For instance, a store selling few high-end products could post big revenue, unlike one shifting lots of budget items for less cash—revenue then misrepresents promotion success in driving buys. Since promotions target heightened customer purchasing, items sold captures response directly, offering a truer effectiveness measure.Key ML principle: target variables must match business objectives.Wrong selections in real projects cause deceptive forecasts and bad choices; focus on outcomes the business truly wants to maximize.

## B1(c) Alternative Modelling Strategy.

Rather than a single model across all 50 stores, adopt a segmented or hierarchical modeling strategy.
Start by grouping stores by location type:
-Urban
-Semi-urban

Rural
Then build distinct models per segment.
This works well since customer behaviors and promotion reactions differ by area—urban shoppers might favor loyalty perks, while rural ones prefer straight discounts.A one-size-fits-all global model overlooks these location-specific traits, hurting accuracy.Segmentation lets models capture group-unique patterns for sharper predictions and smarter campaign choices.It boosts performance by addressing store segment heterogeneity.


### B2 Data and EDA Strategy.

## B2(a) Joining the Data Tables

Constructing the Modeling Dataset.
Merge the four tables via shared keys like store_id, promotion_id, and transaction_date.

Joining Approach
1. Use Transactions table as the base (holds sales data).

2. Link store attributes on store_id for details like:
-location type
-store size
-competition density

3. Attach promotion details via promotion_id adding:
-promotion type
-promotion duration

4. Connect calendar table using transaction_date for:
-weekend flag
-festival flag
-month/season info

Final Dataset Structure:
One row = one store × one month × one promotion
Each row captures a store's performance for a single monthly campaign.

Pre-Modeling Aggregations:
Roll up transaction data to store-month features:

-total items sold
-total sales volume
-total footfall
-average basket size
-number of transactions
-average spend per customer
These summaries cut noise and yield strong predictors of promotion success.

### B2(b) Exploratory Data Analysis (EDA).

Prior to modeling, I'd conduct exploratory data analysis to uncover data patterns and spot valuable features.

1. Items Sold Distribution
Create a histogram of items_sold to check for normal distribution or skewness.
-Reveals:
-skewness levels

outlier extremes
Heavy skew? Apply log(items_sold) transformation for better model results.

2. Promotion Impact by Type
Boxplot of items_sold across promotion types.

Shows:
-top-performing promotions
-performance variation

Divergent results? Add interaction terms like promotion × store type.

3. Numeric Correlations
Correlation heatmap for variables including:
-footfall
-basket size
-competition density
-items sold

Identifies:
-key predictors
-multicollinearity issues
Drop or regularize highly correlated pairs.

4. Location-Based Sales
Bar chart of average items_sold by:

-urban stores
-semi-urban stores
-rural stores

Highlights location's role in promotion success.
Big gaps? Build location-tailored models or interactions.

5. Time Patterns
Line plot of monthly average items_sold trends.
Detects:

-seasonal cycles
-festival impacts
-monthly shifts
Informs time features like month, quarter, or holiday flags.

### B2(c) Handling Promotion Imbalance.

When 80% of transactions lack promotions, models risk bias toward non-promo patterns, struggling to gauge true promotion lift.

Consequences include:
-weak accuracy on promo transactions
-understated promotion impact
-skew toward dominant no-promo class

Mitigation steps:

1. Balance training data
Oversample promo periods or undersample non-promo ones.

2. Add promo flags
Include features explicitly separating promo vs. non-promo cases.

3. Segmented evaluation
Assess accuracy separately for promo and non-promo periods.

This ensures the model captures promotion effects without majority-class dominance.


## B3. Model Evaluation and Deployment.

### B3(a) Train-Test Split and Evaluation Metrics

Given the monthly store data spanning time, I'd apply a chronological train-test split.

Split Approach
Sort by month, then divide sequentially:
Training: First 2.5 years

Test: Final 6 months
Models train on historical promotions, testing against future periods.

Dangers of Random Splitting
Random mixes eras past and future across sets, causing data leakage—models glimpse future patterns in training. Real-world predictions lack future info. Time-evolving promotions and behaviors demand temporal integrity for valid assessment.

Key Metrics
1. MAE (Mean Absolute Error)
Average prediction error in items sold (e.g., MAE=20 means ±20 items typically). Business-friendly and intuitive.

2. RMSE (Root Mean Squared Error)
Hits big errors harder than MAE, critical for avoiding major promotion/inventory missteps.

3. R² Score
Shows variance in items sold explained by model—higher means better capture of store-promo dynamics.
These cover accuracy, error severity, and explanatory power.


### B3(b) Explaining Different Recommendations Using Feature Importance.

When the model suggests varying promotions for one store across months, I'd apply feature importance analysis to pinpoint driving factors.

Influencers might include:
-month/season
-festival timing
-projected footfall
-demand patterns
-past promo results

December's holiday rush boosts traffic, favoring Loyalty Points Bonus for repeat business. March's slack demand suits Flat Discount for quick sales spikes.

Dig deeper with:
-feature importance rankings
-SHAP values
-partial dependence plots
-These reveal variables steering monthly promo picks.
-Explaining to Marketing

Translate into business language:
"December's high footfall + holiday demand led to Loyalty Points Bonus for more repeats. March's low demand favored Flat Discount for instant sales lift."

Links predictions to relatable factors, building team confidence.

### B3(c) Deployment and Monitoring Process

Implement as an automated monthly prediction workflow post-training.

1. Serialize Pipeline
Store full preprocessing/model pipeline via joblib/pickle.

Ensures:
-uniform transformations
-no retraining needed for reuse

2. New Data Processing
-Gather current monthly inputs:
-store characteristics
-promotion candidates
-footfall forecasts
-date features
Transform identically to training, predict items sold across promos/stores, select peak performer.

3. Recommendation Generation
Per store, evaluate:
-Flat Discount
-BOGO
-Free Gift
-Category Offer
-Loyalty Bonus
Assign top scorer as suggestion.

4. Ongoing Monitoring
Monthly: Actuals vs. predictions via:
-MAE
-RMSE
-avg uplift metrics
Detects performance drift.

5. Retraining Signals
Rising errors may reflect:
-behavior evolution
-competitor changes
-seasonality
-promotion saturation
Trigger retraining on latest data for sustained accuracy.