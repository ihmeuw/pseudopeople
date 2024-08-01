[1mdiff --git a/pyproject.toml b/pyproject.toml[m
[1mindex 01f327b..9c2b745 100644[m
[1m--- a/pyproject.toml[m
[1m+++ b/pyproject.toml[m
[36m@@ -19,7 +19,6 @@[m [mexclude = [[m
 disable_error_code = [[m
     "assignment",[m
     "attr-defined",[m
[31m-    "call-overload",[m
     "has-type",[m
     "import-untyped",[m
     "index",[m
[1mdiff --git a/src/pseudopeople/configuration/generator.py b/src/pseudopeople/configuration/generator.py[m
[1mindex edd74e6..1b0f04d 100644[m
[1m--- a/src/pseudopeople/configuration/generator.py[m
[1m+++ b/src/pseudopeople/configuration/generator.py[m
[36m@@ -219,7 +219,7 @@[m [mdef _format_misreport_age_perturbations([m
         if not user_perturbations:[m
             continue[m
         formatted = {}[m
[31m-        default_perturbations: dict[int, float] = default_config[dataset_schema][[m
[32m+[m[32m        default_perturbations: dict[int, float] = default_config[dataset_schema][  # type: ignore[call-overload][m
             Keys.COLUMN_NOISE[m
         ]["age"][NOISE_TYPES.misreport_age.name][Keys.POSSIBLE_AGE_DIFFERENCES][m
         # Replace default configuration with 0 probabilities[m
[1mdiff --git a/src/pseudopeople/configuration/validator.py b/src/pseudopeople/configuration/validator.py[m
[1mindex 8ebf733..f406090 100644[m
[1m--- a/src/pseudopeople/configuration/validator.py[m
[1m+++ b/src/pseudopeople/configuration/validator.py[m
[36m@@ -312,7 +312,7 @@[m [mdef validate_noise_level_proportions([m
     dataset_noise_proportions = dataset_proportions.loc[[m
         (dataset_proportions["state"] == state) & (dataset_proportions["year"] == year)[m
     ][m
[31m-[m
[32m+[m[41m    [m
     # If there is no data for a queried dataset, we want the user's to hit the correct error that there[m
     # is no data available so we do not throw an error here.[m
     if not dataset_noise_proportions.empty:[m
[36m@@ -327,7 +327,7 @@[m [mdef validate_noise_level_proportions([m
                 # Note: Using pd.isnull here and above because np.isnan does not work on strings[m
                 if NOISE_TYPES.duplicate_with_guardian in dataset_schema.row_noise_types:[m
                     # Config level for guardian duplication group[m
[31m-                    config_noise_level = configuration_tree[row["dataset"]][Keys.ROW_NOISE][[m
[32m+[m[32m                    config_noise_level = configuration_tree[row["dataset"]][Keys.ROW_NOISE][  # type: ignore[call-overload][m
                         NOISE_TYPES.duplicate_with_guardian.name[m
                     ][row["noise_type"]][m
                     entity_type = Keys.ROW_NOISE[m
[36m@@ -337,7 +337,7 @@[m [mdef validate_noise_level_proportions([m
                     continue[m
             else:[m
                 # Config level for each column noise type[m
[31m-                config_noise_level = configuration_tree[row["dataset"]][Keys.COLUMN_NOISE][[m
[32m+[m[32m                config_noise_level = configuration_tree[row["dataset"]][Keys.COLUMN_NOISE][  # type: ignore[call-overload][m
                     row["column"][m
                 ][row["noise_type"]][Keys.CELL_PROBABILITY][m
                 entity_type = Keys.COLUMN_NOISE[m
[1mdiff --git a/src/pseudopeople/dataset.py b/src/pseudopeople/dataset.py[m
[1mindex c194805..e385c13 100644[m
[1m--- a/src/pseudopeople/dataset.py[m
[1m+++ b/src/pseudopeople/dataset.py[m
[36m@@ -110,7 +110,7 @@[m [mclass Dataset:[m
                     and noise_type.name in noise_configuration.row_noise[m
                 ):[m
                     # Apply row noise[m
[31m-                    row_noise_configuration: LayeredConfigTree = noise_configuration[[m
[32m+[m[32m                    row_noise_configuration: LayeredConfigTree = noise_configuration[  # type: ignore[call-overload][m
                         Keys.ROW_NOISE[m
                     ][noise_type.name][m
                     noise_type(self, row_noise_configuration)[m
[1mdiff --git a/src/pseudopeople/noise_functions.py b/src/pseudopeople/noise_functions.py[m
[1mindex 1ae9f6b..7ca4c1b 100644[m
[1m--- a/src/pseudopeople/noise_functions.py[m
[1m+++ b/src/pseudopeople/noise_functions.py[m
[36m@@ -219,28 +219,28 @@[m [mdef duplicate_with_guardian([m
             duplicated_rows.append(noised_group_df)[m
 [m
     if duplicated_rows:[m
[31m-        duplicated_rows = pd.concat(duplicated_rows)[m
[32m+[m[32m        duplicated_rows_df: pd.DataFrame = pd.concat(duplicated_rows)[m
         # Update relationship to reference person for duplicated simulants based on housing type[m
[31m-        duplicated_rows["relationship_to_reference_person"] = duplicated_rows[[m
[32m+[m[32m        duplicated_rows_df["relationship_to_reference_person"] = duplicated_rows_df[[m
             "housing_type"[m
         ].map(HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP)[m
 [m
         # Clean columns[m
[31m-        duplicated_rows = duplicated_rows[dataset.data.columns][m
[32m+[m[32m        duplicated_rows_df = duplicated_rows_df[dataset.data.columns][m
 [m
         # Add duplicated rows to the original data and make sure that households[m
         # are grouped together by sorting by date and household_id[m
         # todo if this index is a RangeIndex, we can do concat with ignore_index=True[m
         index_start_value = dataset.data.index.max() + 1[m
[31m-        duplicated_rows.index = range([m
[31m-            index_start_value, index_start_value + len(duplicated_rows)[m
[32m+[m[32m        duplicated_rows_df.index = range([m
[32m+[m[32m            index_start_value, index_start_value + len(duplicated_rows_df)[m
         )[m
         # Note: This is where we would sort the data by year and household_id but[m
         # we ran into runtime issues. It may be sufficient to do it here since this[m
         # would be sorting at the shard level and not the entire dataset.[m
[31m-        data_with_duplicates = pd.concat([dataset.data, duplicated_rows])[m
[32m+[m[32m        data_with_duplicates = pd.concat([dataset.data, duplicated_rows_df])[m
 [m
[31m-        duplicated_rows_missing = dataset.is_missing(duplicated_rows)[m
[32m+[m[32m        duplicated_rows_missing = dataset.is_missing(duplicated_rows_df)[m
         missingess_with_duplicates = pd.concat([m
             [dataset.missingness, duplicated_rows_missing][m
         ).reindex(data_with_duplicates.index)[m
[1mdiff --git a/tests/integration/conftest.py b/tests/integration/conftest.py[m
[1mindex b8de4ee..e0933a7 100644[m
[1m--- a/tests/integration/conftest.py[m
[1m+++ b/tests/integration/conftest.py[m
[36m@@ -144,7 +144,7 @@[m [mdef config():[m
     # This is because we want to be able to compare the noised and unnoised data[m
     # and a big assumption we make is that simulant_id and household_id are the[m
     # truth decks in our datasets.[m
[31m-    config[DATASET_SCHEMAS.census.name][Keys.ROW_NOISE][[m
[32m+[m[32m    config[DATASET_SCHEMAS.census.name][Keys.ROW_NOISE][  # type: ignore[call-overload][m
         NOISE_TYPES.duplicate_with_guardian.name[m
     ] = {[m
         Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.0,[m
[1mdiff --git a/tests/unit/test_row_noise.py b/tests/unit/test_row_noise.py[m
[1mindex 829c2b3..bffc6fa 100644[m
[1m--- a/tests/unit/test_row_noise.py[m
[1m+++ b/tests/unit/test_row_noise.py[m
[36m@@ -34,7 +34,7 @@[m [mdef dummy_data():[m
 [m
 [m
 def test_omit_row(dummy_data, fuzzy_checker: FuzzyChecker):[m
[31m-    config: LayeredConfigTree = get_configuration()[DATASET_SCHEMAS.tax_w2_1099.name][Keys.ROW_NOISE][[m
[32m+[m[32m    config: LayeredConfigTree = get_configuration()[DATASET_SCHEMAS.tax_w2_1099.name][Keys.ROW_NOISE][  # type: ignore[call-overload][m
         NOISE_TYPES.omit_row.name[m
     ][m
     dataset = Dataset(DATASET_SCHEMAS.tax_w2_1099, dummy_data, 0)[m
