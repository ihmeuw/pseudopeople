**0.6.0 - 04/19/23**
 - Update documentation (landing page, datasets section, quickstart)
 - Update zipcode miswriting function to act on each digit independently
 - Modify config key names
 - Update sample datsets to include all GQ types
 - Scale household survey data to account for oversampling
 - Implement user config value validation
 - Change the term "Form" to "Dataset" throughout
 - Update the default config values
 - Change "american_communities_survey" to "american_community_survey"
 - Implement config interface and get_config function
 - Add a github issues template

**0.5.1 - 04/14/23**
 - Formatting of noised dates implemented
 - Moved from pd.NA to np.nan
 - Added validation of user-supplied configuration
 - Changed 'row_noise_level' to 'probability'
 - Improved logging and added a noising progress bar
 - Set default logging level to 'INFO', configurable with 'verbose' flag

**0.5.0 - 04/13/23**
 - Bugfix to apply incorrect selection noising at the expected probability
 - Implement the omission noise function
 - Implement schema for output columns and their dtypes
 - Implement a year filter to the form generation functions
 - Support a path to data root directory as form generation function argument
 - Update documentation
 
 **0.4.0 - 04/11/23**
 - Generate default configuration instead of maintaining a static file
 - Read sample data if no data argument is provided
 - Update sample datasets

**0.3.2 - 04/10/23**

 - Update sample datasets

**0.3.1 - 04/10/23**

 - Build docs to readthedocs
 - Implement zipcode miswriting function
 - Implement fake name noise function
 - Add sample data to package
 - Support parquet files

**0.3.0 - 04/04/23**

 - Implement numeric miswriting noise function
 - Implement age miswriting noise function
 - Implement additional forms: ACS, CPS, WIC, and SSA
 - Read data in from HDF files instead of CSV files

**0.2.1 - 03/31/23**

 - Fix bug preventing generation of W2/1099 forms

**0.2.0 - 03/31/23**

 - Implemented W2/1099 forms
 - Implemented typographic noise function
 - Implemented incorrect selection noise function

**0.1.0 - 03/23/23**

 - Initial release
 - Implemented generate_decennial_census with missing data noise function
 - Implemented custom user configuration override