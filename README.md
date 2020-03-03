# Project description
The project presents the solution of machine learning contest organized at Skoltech by Softline company. The solution was provided by Skoltech students Egor Gladin, Semyon Glushkov and Jamil Zakirov and was **awarded the 2nd place** (see [kaggle profile](https://www.kaggle.com/egorgladin/competitions)).<br>
**Contest task:** predict weekly income of [softline online shop](https://store.softline.ru/) for multiple categories.<br>
**Dataset:** sales in the period from 2015-11-01 till 2019-12-31 excluding 11 weeks for which the income should be predicted:<br>
from 2019-03-25 till 2019-03-31,<br>
from 2019-04-22 till 2019-04-28,<br>
from 2019-05-27 till 2019-06-02,<br>
from 2019-06-24 till 2019-06-30,<br>
from 2019-07-22 till 2019-07-28,<br>
from 2019-08-26 till 2019-09-01,<br>
from 2019-09-23 till 2019-09-29,<br>
from 2019-10-21 till 2019-10-27,<br>
from 2019-11-25 till 2019-12-01,<br>
from 2019-12-16 till 2019-12-22,<br>
from 2020-01-06 till 2020-01-12.<br>
For each of these weeks, **the following values should predicted:** *full_discount_price* and *full_price* aggregated for each of 38 popular stores (indicated by *market_id*), for each of 46 popular categories (indicated by *category_id*) and for each of 43 popular vendors (indicated by *vendors_id*) and also total income (including less popular stores, categories and vendors).<br>
Thus, we have 2(38 + 46 + 43 + 1) = 256 values to predict for each missing week (multiplication by 2 is because we predict both *full_discount_price* and *full_price*).<br>
**Metric:** RMSE
