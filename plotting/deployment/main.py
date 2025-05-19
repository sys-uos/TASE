from TASE.plotting.scripts.evaluation import make_plots_of_species_heatmaps, make_plots_about_impact_of_interference, \
    make_plots_about_methodological_challenges, make_plots_about_impact_of_timeintervals
from TASE.plotting.scripts.formalization import make_plots_for_formalization
from TASE.plotting.scripts.problem_issue import make_plots_for_problem_issue
from TASE.plotting.scripts.range_analysis import make_plots_for_range_analysis
from TASE.plotting.deployment.analysis import apply_tase_for_all_20230603
from TASE.plotting.deployment.parsing import parse_data_from_20230603

def evaluation_of_deployment_20230603():
    # --- Parse data from the deployment --- #
    parse_data_from_20230603()
    # exit(0)

    # --- Apply TASE on the data --- #
    apply_tase_for_all_20230603()
    # exit(0)

    # --- Make plotting for the paper --- #
    make_plots_for_problem_issue()
    make_plots_for_formalization()
    make_plots_for_range_analysis()
    make_plots_of_species_heatmaps()
    make_plots_about_impact_of_timeintervals()
    make_plots_about_impact_of_interference()
    make_plots_about_methodological_challenges()
    exit(0)
