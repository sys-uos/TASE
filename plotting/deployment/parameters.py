from TASE.src.core.params import Parameters


def get_TASE_ParameterSet(spec):
    params = [
        Parameters(
            threshold_R=0.5,
            threshold_B=0.1,
            TS_delta=0.2,
            e_threshold_meter=100,
            threshold_T=spec.max_root_2_leaf_distance(),
            e_delta=0.2,
        ),
        Parameters(
            threshold_R=0.6,
            threshold_B=0.1,
            TS_delta=0.2,
            e_threshold_meter=100,
            threshold_T=spec.max_root_2_leaf_distance(),
            e_delta=0.2,
        ),
        Parameters(
            threshold_R=0.7,
            threshold_B=0.1,
            TS_delta=0.2,
            e_threshold_meter=100,
            threshold_T=spec.max_root_2_leaf_distance(),
            e_delta=0.2,
        ),
        Parameters(
            threshold_R=0.8,
            threshold_B=0.1,
            TS_delta=0.2,
            e_threshold_meter=100,
            threshold_T=spec.max_root_2_leaf_distance(),
            e_delta=0.2,
        )
    ]

    return params