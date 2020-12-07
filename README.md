# myfinanceai

You can use this app to retrieve and use machine learning algorithms out of the bat to predict the prices of selected financial instruments.

Due to controversal opinion about sharing symbol data, the metadata file is excluded from this repository. Anyways one may use the retrieval methods used in this code for any symbol available through Yahoo Finance.

The metadata schema is the following:

|    Name  |Symbol| Currency|      Country           |
-------------------------------
|Apple Inc.|  AAPL| USD     |United States of America|

Symbols can be found e.g. at
https://finance.yahoo.com/lookup/all?s=a

They are used to fetch data from Yahoo Finance with pandas datareader.

The CSS files have been removed, you may use other publicly available or custom css.
In this app I use these publicly open files
https://codepen.io/chriddyp/pen/bWLwgP.css
https://codepen.io/chriddyp/pen/brPBPO.css
These can be called directly from code or by inserting them into "assets" folder.


