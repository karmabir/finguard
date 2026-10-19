import pytest

from finguard import data, model, tools


def test_pipeline_runs():
    data.build_db()
    clf, info = model.train_eval()
    assert 0 <= info["test_pr_auc"] <= 1


def test_select_guard():
    data.build_db()
    assert tools.run_select("select count(*) as n from transactions")[0]["n"] > 0
    for bad in ["drop table transactions", "select 1; delete from transactions", "update transactions set isFraud=0"]:
        with pytest.raises(ValueError):
            tools.run_select(bad)
