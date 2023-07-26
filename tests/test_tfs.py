import numpy as np
from qsipy import tfs


def relative_deviation(array_tested, array_tabulated):
    return np.abs(array_tested - array_tabulated) / array_tabulated


def test_ev_vhd_perfect_qe(RELATIVE_DIFF=1e-14):
    # check value at zero
    assert tfs.ev_vhd_perfect_qe(0, 2) == 0
    assert tfs.ev_vhd_perfect_qe(0, 10) == 0
    assert tfs.ev_vhd_perfect_qe(0, 100) == 0

    # check nonzero values
    phi_inputs = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    tabulated_N_equal_2 = np.array(
        [
            0.0000999966667111108,
            0.00249791736098712,
            0.00996671107937919,
            0.22984884706593,
            0.708073418273571,
        ]
    )
    tabulated_N_equal_100 = np.array(
        [
            0.127495750056666,
            3.18484463525857,
            12.7075566262085,
            293.057280009061,
            902.793608298803,
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.ev_vhd_perfect_qe(phi_inputs, 2),
            array_tabulated=tabulated_N_equal_2,
        )
        < RELATIVE_DIFF
    )
    assert check.all()

    check = (
        relative_deviation(
            array_tested=tfs.ev_vhd_perfect_qe(phi_inputs, 100),
            array_tabulated=tabulated_N_equal_100,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_ev_vhd_squared_perfect_qe(RELATIVE_DIFF=1e-14):
    # check value at zero
    assert tfs.ev_vhd_squared_perfect_qe(0, 2) == 0
    assert tfs.ev_vhd_squared_perfect_qe(0, 10) == 0
    assert tfs.ev_vhd_squared_perfect_qe(0, 100) == 0

    # check nonzero values
    phi_inputs = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    tabulated_N_equal_2 = np.array(
        [
            0.0000999966667111108,
            0.00249791736098712,
            0.00996671107937919,
            0.22984884706593,
            0.708073418273571,
        ]
    )
    tabulated_N_equal_100 = np.array(
        [
            0.151859375755396,
            18.3877644433019,
            254.740570920523,
            129015.873012571,
            1222498.37615172,
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.ev_vhd_squared_perfect_qe(phi_inputs, 2),
            array_tabulated=tabulated_N_equal_2,
        )
        < RELATIVE_DIFF
    )
    assert check.all()

    check = (
        relative_deviation(
            array_tested=tfs.ev_vhd_squared_perfect_qe(phi_inputs, 100),
            array_tabulated=tabulated_N_equal_100,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_fluctuations_vhd_perfect_qe(RELATIVE_DIFF=1e-14):
    # check value at zero
    assert tfs.fluctuations_vhd_perfect_qe(0, 2) == 0
    assert tfs.fluctuations_vhd_perfect_qe(0, 10) == 0
    assert tfs.fluctuations_vhd_perfect_qe(0, 100) == 0

    # check nonzero values
    phi_inputs = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    tabulated_N_equal_2 = np.array(
        [
            0.00999933334666654,
            0.0499167083234141,
            0.0993346653975306,
            0.420735492403948,
            0.454648713412841,
        ]
    )
    tabulated_N_equal_100 = np.array(
        [
            0.368244768425682,
            2.87132880258715,
            9.65704797089816,
            207.685588441428,
            638.327562436834,
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.fluctuations_vhd_perfect_qe(phi_inputs, 2),
            array_tabulated=tabulated_N_equal_2,
        )
        < RELATIVE_DIFF
    )
    assert check.all()

    check = (
        relative_deviation(
            array_tested=tfs.fluctuations_vhd_perfect_qe(phi_inputs, 100),
            array_tabulated=tabulated_N_equal_100,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_phase_uncertainty_vhd_perfect_qe(RELATIVE_DIFF=1e-14):
    # check N=2 => phase uncertainty = 1/2
    delta_phi = tfs.phase_uncertainty_vhd_perfect_qe(0, 2)
    assert relative_deviation(delta_phi, 0.5) < RELATIVE_DIFF

    delta_phi = tfs.phase_uncertainty_vhd_perfect_qe(0.1, 2)
    assert relative_deviation(delta_phi, 0.5) < RELATIVE_DIFF

    delta_phi = tfs.phase_uncertainty_vhd_perfect_qe(1, 2)
    assert relative_deviation(delta_phi, 0.5) < RELATIVE_DIFF

    phi_inputs = np.array([0, 0.01, 0.05, 0.1, 0.5, 1.0])
    tabulated_N_equal_100 = np.array(
        [
            0.0140028008402801,
            0.0144419340871611,
            0.02255780344803,
            0.0381244313904991,
            0.19357846027015,
            0.550588898426396,
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.phase_uncertainty_vhd_perfect_qe(phi_inputs, 100),
            array_tabulated=tabulated_N_equal_100,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_ev_vhd_finite_qe(RELATIVE_DIFF=1e-14):
    # check various values
    phi_inputs = np.array([0, 0.01, 0.05, 0.1, 0.5, 1.0])
    N_inputs = np.array([2, 100])
    eta_inputs = np.array([0.5, 0.75, 0.9, 0.95])
    phi, N, eta = np.meshgrid(phi_inputs, N_inputs, eta_inputs, indexing="ij")
    tabulated = np.array(
        [
            [[0.125, 0.09375, 0.045, 0.02375], [6.25, 4.6875, 2.25, 1.1875]],
            [
                [
                    0.125024999166678,
                    0.093806248125025,
                    0.045080997300036,
                    0.0238402469917068,
                ],
                [6.28187393751417, 4.75921635940687, 2.3532715575459, 1.30256491442614],
            ],
            [
                [
                    0.125624479340247,
                    0.0951550785155553,
                    0.0470233130623996,
                    0.0260043704182909,
                ],
                [
                    7.04621115881464,
                    6.47897510733295,
                    4.82972415455945,
                    4.06182228332087,
                ],
            ],
            [
                [
                    0.127491677769845,
                    0.0993562749821508,
                    0.0530730359742971,
                    0.0327449567491397,
                ],
                [
                    9.42688915655211,
                    11.8355006022423,
                    12.5431208672289,
                    12.6560698551531,
                ],
            ],
            [
                [
                    0.182462211766483,
                    0.223039976474586,
                    0.231177566123403,
                    0.231188584477002,
                ],
                [
                    79.5143200022652,
                    169.532220005097,
                    239.626396807339,
                    265.671695208178,
                ],
            ],
            [
                [
                    0.302018354568393,
                    0.492041297778884,
                    0.618539468801593,
                    0.662786259991898,
                ],
                [231.948402074701, 512.508904668077, 733.512822722031, 815.95873148967],
            ],
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.ev_vhd_finite_qe(phi, N, eta),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_ev_vhd_squared_finite_qe(RELATIVE_DIFF=1e-11):
    # check various values
    phi_inputs = np.array([0, 0.01, 0.05, 0.1, 0.5, 1.0])
    N_inputs = np.array([2, 100])
    eta_inputs = np.array([0.5, 0.75, 0.9, 0.95])
    phi, N, eta = np.meshgrid(phi_inputs, N_inputs, eta_inputs, indexing="ij")
    tabulated = np.array(
        [
            [
                [
                    0.0312500000000000,
                    0.0234375000000000,
                    0.0112500000000000,
                    0.00593750000000000,
                ],
                [
                    116.406250000000,
                    65.7714843750000,
                    15.4462500000000,
                    4.44273437500000,
                ],
            ],
            [
                [
                    0.0312749991666778,
                    0.0234937481250250,
                    0.0113309973000360,
                    0.00602774699170678,
                ],
                [
                    117.611013867766,
                    67.8275916940026,
                    16.9317892386991,
                    5.38108437981095,
                ],
            ],
            [
                [
                    0.0318744793402468,
                    0.0248425785155553,
                    0.0132733130623996,
                    0.00819187041829087,
                ],
                [
                    147.413403733255,
                    121.750790973711,
                    62.1303604054381,
                    39.7698852035089,
                ],
            ],
            [
                [
                    0.0337416777698448,
                    0.0290437749821508,
                    0.0193230359742971,
                    0.0149324567491397,
                ],
                [
                    251.460879053237,
                    346.517010005605,
                    320.715220619166,
                    293.127996092818,
                ],
            ],
            [
                [
                    0.0887122117664825,
                    0.152727476474586,
                    0.197427566123403,
                    0.213376084477002,
                ],
                [
                    10927.3103133706,
                    45502.8522149016,
                    87848.3517787025,
                    106961.225754394,
                ],
            ],
            [
                [
                    0.208268354568393,
                    0.421728797778884,
                    0.584789468801593,
                    0.644973759991898,
                ],
                [
                    84986.2448372840,
                    401090.897644337,
                    811910.177924075,
                    1.00150559104375 * 1e6,
                ],
            ],
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.ev_vhd_squared_finite_qe(phi, N, eta),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_fluctuations_vhd_finite_qe(RELATIVE_DIFF=1e-11):
    # check various values
    phi_inputs = np.array([0, 0.01, 0.05, 0.1, 0.5, 1.0])
    N_inputs = np.array([2, 100])
    eta_inputs = np.array([0.5, 0.75, 0.9, 0.95])
    phi, N, eta = np.meshgrid(phi_inputs, N_inputs, eta_inputs, indexing="ij")
    tabulated = np.array(
        [
            [
                [
                    0.1250000000000000,
                    0.1210307295689818,
                    0.09604686356149273,
                    0.07330373455697875,
                ],
                [
                    8.794529549668930,
                    6.618068307671053,
                    3.222382658841125,
                    1.741429908150196,
                ],
            ],
            [
                [
                    0.1250749725166870,
                    0.1212193711323864,
                    0.09642977228257955,
                    0.07388768243138496,
                ],
                [
                    8.840196474114488,
                    6.721417360821791,
                    3.375485478437818,
                    1.919481446515430,
                ],
            ],
            [
                [
                    0.1268580684495026,
                    0.1256506647346275,
                    0.1051766185567644,
                    0.08669280902957979,
                ],
                [
                    9.887583731056415,
                    8.931610858757285,
                    6.229295706281984,
                    4.824052750771590,
                ],
            ],
            [
                [
                    0.1322405001097427,
                    0.1384633727879761,
                    0.1284768026795814,
                    0.1177294549237220,
                ],
                [
                    12.75126032529009,
                    14.36794820076715,
                    12.78222748699359,
                    11.53047665598879,
                ],
            ],
            [
                [
                    0.2354140034997192,
                    0.3209059759007332,
                    0.3794528943685406,
                    0.3999098934566662,
                ],
                [
                    67.85855309353429,
                    129.4669015426145,
                    174.4349211936475,
                    190.7348319516069,
                ],
            ],
            [
                [
                    0.3421304839855598,
                    0.4238209044619621,
                    0.4496647577209451,
                    0.4535285366521606,
                ],
                [
                    176.5961030495206,
                    372.0558026429172,
                    523.3250584736353,
                    579.4108555675496,
                ],
            ],
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.fluctuations_vhd_finite_qe(phi, N, eta),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_phase_uncertainty_vhd_finite_qe(RELATIVE_DIFF=1e-11):
    # check various values
    phi_inputs = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    N_inputs = np.array([2, 100])
    eta_inputs = np.array([0.5, 0.75, 0.9, 0.95])

    # check divergence in (phi = 0)
    N, eta = np.meshgrid(N_inputs, eta_inputs, indexing="ij")
    check = tfs.phase_uncertainty_vhd_finite_qe(0, N, eta) == np.inf
    assert check.all()

    phi, N, eta = np.meshgrid(phi_inputs, N_inputs, eta_inputs, indexing="ij")
    tabulated = np.array(
        [
            [
                [
                    25.0166622474653,
                    10.7757735823056,
                    5.95285192802972,
                    4.09377332447440,
                ],
                [
                    1.38678993694903,
                    0.468626787013471,
                    0.163433092292665,
                    0.0834114135209242,
                ],
            ],
            [
                [
                    5.08278981969643,
                    2.23751692600539,
                    1.30064342774047,
                    0.962187986613910,
                ],
                [
                    0.310716306930982,
                    0.124744425442541,
                    0.0604182037385300,
                    0.0419931756336350,
                ],
            ],
            [
                [
                    2.66252470032541,
                    1.23902922606887,
                    0.798378572106216,
                    0.656609397021293,
                ],
                [
                    0.201359484131754,
                    0.100839622957817,
                    0.0622989188591163,
                    0.0504381369638582,
                ],
            ],
            [
                [
                    1.11905939836280,
                    0.677978828756237,
                    0.556716003136933,
                    0.526593861646706,
                ],
                [
                    0.252996932961697,
                    0.214529439168280,
                    0.200723898933259,
                    0.196985142180394,
                ],
            ],
            [
                [
                    1.50503223210440,
                    0.828617087744820,
                    0.610517152937157,
                    0.552651618190229,
                ],
                [
                    0.609291276555532,
                    0.570518075559945,
                    0.557276032123767,
                    0.553762152696507,
                ],
            ],
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.phase_uncertainty_vhd_finite_qe(phi, N, eta),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_optimal_phi_vhd(RELATIVE_DIFF=1e-14):
    # check various values
    N_inputs = np.array([2, 50, 100])
    eta_inputs = np.array([0.5, 0.75, 0.9, 0.95])

    N, eta = np.meshgrid(N_inputs, eta_inputs, indexing="ij")
    tabulated = np.array(
        [
            [
                0.523598775598299,
                0.506068904985568,
                0.514643037152054,
                0.518929528582268,
            ],
            [
                0.252417483653861,
                0.154889468715136,
                0.0926411347623607,
                0.0655945997013353,
            ],
            [
                0.188182696924450,
                0.112341469336886,
                0.0660599460726298,
                0.0461448377619527,
            ],
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.optimal_phi_vhd(N, eta),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_phase_uncertainty_at_optimal_phi_vhd(RELATIVE_DIFF=1e-12):
    # perfect case
    N_inputs = np.array([2, 10, 20, 50, 100])
    tabulated = np.array(
        [
            0.5000000000000000000,
            0.1290994448735805628,
            0.06741998624632420862,
            0.02773500981126145610,
            0.01400280084028009804,
        ]
    )
    check = (
        relative_deviation(
            array_tested=tfs.phase_uncertainty_at_optimal_phi_vhd(N_inputs),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()

    # check various values with eta != 1
    N_inputs = np.array([2, 50, 100])
    eta_inputs = np.array([0.5, 0.75, 0.9, 0.95])

    N, eta = np.meshgrid(N_inputs, eta_inputs, indexing="ij")
    tabulated = np.array(
        [
            [
                1.118033988749895,
                0.6779558757222977,
                0.5566690173508563,
                0.5265558707122137,
            ],
            [
                0.2499466370434959,
                0.1424044238064738,
                0.0849110052758587,
                0.06206832970494668,
            ],
            [
                0.1751567662226774,
                0.1003804211940519,
                0.05892753707014577,
                0.0419103269101814,
            ],
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.phase_uncertainty_at_optimal_phi_vhd(N, eta),
            array_tabulated=tabulated,
        )
        < RELATIVE_DIFF
    )
    assert check.all()


def test_asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd():
    N = 10 ** np.linspace(1, 5, 10)

    # test 1
    eta = 0.95
    values = tfs.phase_uncertainty_at_optimal_phi_vhd(N, eta) * np.sqrt(N * eta)
    limit = tfs.asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd(eta)

    # difference between limit and computed values
    square_diff = (values - np.full_like(values, limit)) ** 2
    # checking that the difference gets smaller when N increases (study the derivate)
    derivative_square_diff = np.diff(square_diff)
    check = derivative_square_diff < 0
    assert check.all()
    # checking the the computed values reach get arbitrarily close to the limit
    val = tfs.phase_uncertainty_at_optimal_phi_vhd(1e5, eta)
    assert (val * np.sqrt(1e5 * eta) - limit) ** 2 < 1e-9

    # test 2
    eta = 0.75
    values = tfs.phase_uncertainty_at_optimal_phi_vhd(N, eta) * np.sqrt(N * eta)
    limit = tfs.asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd(eta)

    # difference between limit and computed values
    square_diff = (values - np.full_like(values, limit)) ** 2
    # checking that the difference gets smaller when N increases (study the derivate)
    derivative_square_diff = np.diff(square_diff)
    check = derivative_square_diff < 0
    assert check.all()
    # checking the the computed values reach get arbitrarily close to the limit
    val = tfs.phase_uncertainty_at_optimal_phi_vhd(1e5, eta)
    assert (val * np.sqrt(1e5 * eta) - limit) ** 2 < 1e-9

    # test 3
    eta = 0.5
    values = tfs.phase_uncertainty_at_optimal_phi_vhd(N, eta) * np.sqrt(N * eta)
    limit = tfs.asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd(eta)

    # difference between limit and computed values
    square_diff = (values - np.full_like(values, limit)) ** 2
    # checking that the difference gets smaller when N increases (study the derivate)
    derivative_square_diff = np.diff(square_diff)
    check = derivative_square_diff < 0
    assert check.all()
    # checking the the computed values reach get arbitrarily close to the limit
    val = tfs.phase_uncertainty_at_optimal_phi_vhd(1e5, eta)
    assert (val * np.sqrt(1e5 * eta) - limit) ** 2 < 1e-9
