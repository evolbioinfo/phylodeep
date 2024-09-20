import os
import unittest

from modeldeep import modeldeep
from phylodeep import FULL, BDSS, SUMSTATS, BDEI, BD, PROBABILITY_BDEI, PROBABILITY_BDSS, PROBABILITY_BD, \
    INFECTIOUS_PERIOD, X_TRANSMISSION, SS_FRACTION, R, INCUBATION_PERIOD
from phylodeep.checkdeep import checkdeep
from phylodeep.paramdeep import paramdeep

CI_upper = 'ci_97_5_boundary'

CI_lower = 'ci_2_5_boundary'

P = 0.8773057189071051

BD_NWK = os.path.join(os.path.dirname(__file__), 'data', 'BD.huge.nwk')
BDEI_NWK = os.path.join(os.path.dirname(__file__), 'data', 'BDEI.huge.nwk')
BDSS_NWK = os.path.join(os.path.dirname(__file__), 'data', 'BDSS.huge.nwk')


class HugeTreeTest(unittest.TestCase):

    def test_model_selection_FULL_BD(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(BD_NWK, proba_sampling=P, vector_representation=FULL)
        self.assertAlmostEqual(0.017126, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.818495, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BD], places=5)
        self.assertAlmostEqual(0.164379, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDSS], places=5)

    def test_model_selection_FULL_BDEI(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(BDEI_NWK, proba_sampling=P, vector_representation=FULL)
        self.assertAlmostEqual(0.647439, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.338836, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BD], places=5)
        self.assertAlmostEqual(0.013725, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDSS], places=5)

    def test_model_selection_FULL_BDSS(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(BDSS_NWK, proba_sampling=P, vector_representation=FULL)
        self.assertAlmostEqual(0, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BD], places=5)
        self.assertAlmostEqual(1, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDSS], places=5)

    def test_model_selection_SUMSTATS_BD(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(BD_NWK, proba_sampling=P, vector_representation=SUMSTATS)
        print(df_BDEI_vs_BD_vs_BDSS)
        self.assertAlmostEqual(0.016466, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.944058, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BD], places=5)
        self.assertAlmostEqual(0.039476, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDSS], places=5)

    def test_model_selection_SUMSTATS_BDEI(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(BDEI_NWK, proba_sampling=P, vector_representation=SUMSTATS)
        print(df_BDEI_vs_BD_vs_BDSS)
        self.assertAlmostEqual(0.760552, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.197801, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BD], places=5)
        self.assertAlmostEqual(0.041647, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDSS], places=5)

    def test_model_selection_SUMSTATS_BDSS(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(BDSS_NWK, proba_sampling=P, vector_representation=SUMSTATS)
        print(df_BDEI_vs_BD_vs_BDSS)
        self.assertAlmostEqual(0, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(1, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BDSS], places=5)
        self.assertAlmostEqual(0, df_BDEI_vs_BD_vs_BDSS.loc[0, PROBABILITY_BD], places=5)

    def test_estimate_bdss_FULL(self):
        df = paramdeep(BDSS_NWK, proba_sampling=P, vector_representation=FULL, model=BDSS)
        print(df)
        self.assertAlmostEqual(2.7728, df.loc[0, R], places=5)
        self.assertAlmostEqual(6.096797, df.loc[0, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(6.252192, df.loc[0, X_TRANSMISSION], places=5)
        self.assertAlmostEqual(0.081487, df.loc[0, SS_FRACTION], places=5)

    def test_estimate_bdss_SUMSTATS(self):
        df = paramdeep(BDSS_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDSS)
        print(df.iloc[0, :])
        self.assertAlmostEqual(2.685904, df.loc[0, R], places=5)
        self.assertAlmostEqual(6.115654, df.loc[0, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(5.513904, df.loc[0, X_TRANSMISSION], places=5)
        self.assertAlmostEqual(0.107052, df.loc[0, SS_FRACTION], places=5)

    def test_estimate_bdei_FULL(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=FULL, model=BDEI)
        print(df)
        self.assertAlmostEqual(3.425037, df.loc[0, R], places=5)
        self.assertAlmostEqual(6.774225, df.loc[0, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(3.878085, df.loc[0, INCUBATION_PERIOD], places=5)

    def test_estimate_bdei_SUMSTATS(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDEI)
        print(df)
        self.assertAlmostEqual(3.643254, df.loc[0, R], places=5)
        self.assertAlmostEqual(6.899823, df.loc[0, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(3.9644, df.loc[0, INCUBATION_PERIOD], places=5)

    def test_estimate_bd_FULL(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=FULL, model=BD)
        print(df)
        self.assertAlmostEqual(3.024165, df.loc[0, R], places=5)
        self.assertAlmostEqual(6.628268, df.loc[0, INFECTIOUS_PERIOD], places=5)

    def test_estimate_bd_SUMSTATS(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BD)
        print(df)
        self.assertAlmostEqual(3.099353, df.loc[0, R], places=5)
        self.assertAlmostEqual(6.730956, df.loc[0, INFECTIOUS_PERIOD], places=5)

    def test_estimate_bdss_FULL_CI(self):
        df = paramdeep(BDSS_NWK, proba_sampling=P, vector_representation=FULL, model=BDSS, ci_computation=True)
        print(df.iloc[:, 0:])
        self.assertAlmostEqual(2.225275, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.611817, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(5.186102, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(7.300588, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(4.444617, df.loc[CI_lower, X_TRANSMISSION], places=5)
        self.assertAlmostEqual(8.270996, df.loc[CI_upper, X_TRANSMISSION], places=5)
        self.assertAlmostEqual(0.054042, df.loc[CI_lower, SS_FRACTION], places=5)
        self.assertAlmostEqual(0.137394, df.loc[CI_upper, SS_FRACTION], places=5)

    def test_estimate_bdss_SUMSTATS_CI(self):
        df = paramdeep(BDSS_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDSS, ci_computation=True)
        print(df.iloc[:, 0:])
        self.assertAlmostEqual(2.15239, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.45362, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(5.201134, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(7.31562, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(3.993311, df.loc[CI_lower, X_TRANSMISSION], places=5)
        self.assertAlmostEqual(7.602766, df.loc[CI_upper, X_TRANSMISSION], places=5)
        self.assertAlmostEqual(0.068558, df.loc[CI_lower, SS_FRACTION], places=5)
        self.assertAlmostEqual(0.160411, df.loc[CI_upper, SS_FRACTION], places=5)

    def test_estimate_bdei_FULL_CI(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=FULL, model=BDEI, ci_computation=True)
        print(df)
        self.assertAlmostEqual(2.651773, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(4.301801, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(5.518205, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(8.285918, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(2.813453, df.loc[CI_lower, INCUBATION_PERIOD], places=5)
        self.assertAlmostEqual(5.205532, df.loc[CI_upper, INCUBATION_PERIOD], places=5)

    def test_estimate_bdei_SUMSTATS_CI(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDEI, ci_computation=True)
        print(df)
        self.assertAlmostEqual(2.828514, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(4.438676, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(5.626881, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(8.319689, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(2.857359, df.loc[CI_lower, INCUBATION_PERIOD], places=5)
        self.assertAlmostEqual(5.14108, df.loc[CI_upper, INCUBATION_PERIOD], places=5)

    def test_estimate_bd_FULL_CI(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=FULL, model=BD, ci_computation=True)
        print(df)
        self.assertAlmostEqual(2.434932, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.836745, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(5.515483, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(7.953172, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)

    def test_estimate_bd_SUMSTATS_CI(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BD, ci_computation=True)
        print(df)
        self.assertAlmostEqual(2.492101, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.918626, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(5.620326, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(8.046475, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)

    def test_chekdeep_bd_runs(self):
        png = 'BD_a_priori_check_BD_huge.png'
        os.remove(png) if os.path.exists(png) else None
        checkdeep(BD_NWK, model=BD, outputfile_png=png)
        self.assertTrue(os.path.exists(png))
        os.remove(png) if os.path.exists(png) else None

    def test_chekdeep_bdei_runs(self):
        png = 'BDEI_a_priori_check_BDEI_huge.png'
        os.remove(png) if os.path.exists(png) else None
        checkdeep(BDEI_NWK, model=BDEI, outputfile_png=png)
        self.assertTrue(os.path.exists(png))
        os.remove(png) if os.path.exists(png) else None

    def test_chekdeep_bdss_runs(self):
        png = 'BDSS_a_priori_check_BDSS_huge.png'
        os.remove(png) if os.path.exists(png) else None
        checkdeep(BDSS_NWK, model=BDSS, outputfile_png=png)
        self.assertTrue(os.path.exists(png))
        os.remove(png) if os.path.exists(png) else None
