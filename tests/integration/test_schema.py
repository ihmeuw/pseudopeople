import pytest

from pseudopeople.schema_entities import COLUMNS, DATASETS
from tests.integration.conftest import _get_common_datasets, _load_sample_data


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        DATASETS.tax_1040.name,
    ],
)
def test_unnoised_id_cols(dataset_name: str, request):
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    unnoised_id_cols = [COLUMNS.simulant_id.name]
    if dataset_name != DATASETS.ssa.name:
        unnoised_id_cols.append(COLUMNS.household_id.name)
    data = _load_sample_data(dataset_name, request)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original, _ = _get_common_datasets(dataset_name, data, noised_data)
    assert (
        (
            check_original.reset_index()[unnoised_id_cols]
            == check_noised.reset_index()[unnoised_id_cols]
        )
        .all()
        .all()
    )
