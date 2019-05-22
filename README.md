# Machine Learning Models

*Author*: Nicolo Ceneda \
*Contact*: nicolo.ceneda@student.unisg.ch \
*Website*: [www.nicoloceneda.com](http://www.nicoloceneda.com) \
*Institution*: University of St Gallen \
*Course*: Master of Banking and Finance \
*Last update*: 20 May 2019

## Project Structure
<pre>
│
├── 02_perceptron.py                      <--  Implementation of a single layer perceptron.
│                                              
![Decision_boundary_and_training_sample](https://user-images.githubusercontent.com/47401951/58213656-42bb9e80-7cf3-11e9-9745-4079b0714e0e.png)
│ 
├── extract_data.py                       <--  Command line interface to extract and clean trade 
│        │                                     data downloaded from the wrds database.
│        │
│        └── extract_data_functions.py    <--  General functions called in 'extract_data.py'
│
│
├── data_analysis.py                      <--  Analysis of general data.
│
│
├── extract_data_bg.sh                    <--  Wrapper script to execute 'extract_data.py' in 
│                                              'debugging' mode.
│
├── extract_data_sl.sh                    <--  Wrapper script to execute extract_data.py in 
│                                              'symbol_list' mode.
│
└── nasdaq100.xlsx                        <--  List of securities extracted
</pre>

![z_Original_Aggregated](https://user-images.githubusercontent.com/47401951/58116887-b1b6cb80-7bfd-11e9-9457-cca3e8dd3fea.png)
