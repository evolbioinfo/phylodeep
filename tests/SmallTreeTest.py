import os
import unittest

from checkdeep import checkdeep
from phylodeep.modeldeep import modeldeep
from phylodeep.paramdeep import paramdeep
from phylodeep import FULL, SUMSTATS, BDEI, BD, PROBABILITY_BDEI, PROBABILITY_BD, \
    INFECTIOUS_PERIOD, R, INCUBATION_PERIOD

CI_upper = 'ci_97_5_boundary'

CI_lower = 'ci_2_5_boundary'

P = 0.3

BD_NWK = os.path.join(os.path.dirname(__file__), 'data', 'BD.small.nwk')
BDEI_NWK = os.path.join(os.path.dirname(__file__), 'data', 'BDEI.small.nwk')


class SmallTreeTest(unittest.TestCase):

    def test_model_selection_FULL_BD(self):
        df_BD_vs_BDEI = modeldeep(BD_NWK, proba_sampling=P, vector_representation=FULL)
        print(df_BD_vs_BDEI)
        self.assertAlmostEqual(0.045525, df_BD_vs_BDEI.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.954475, df_BD_vs_BDEI.loc[0, PROBABILITY_BD], places=5)

    def test_model_selection_FULL_BDEI(self):
        df_BD_vs_BDEI = modeldeep(BDEI_NWK, proba_sampling=P, vector_representation=FULL)
        print(df_BD_vs_BDEI)
        self.assertAlmostEqual(0.201486, df_BD_vs_BDEI.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.798514, df_BD_vs_BDEI.loc[0, PROBABILITY_BD], places=5)

    def test_model_selection_SUMSTATS_BD(self):
        df_BD_vs_BDEI = modeldeep(BD_NWK, proba_sampling=P, vector_representation=SUMSTATS)
        print(df_BD_vs_BDEI)
        self.assertAlmostEqual(0.113622, df_BD_vs_BDEI.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.886378, df_BD_vs_BDEI.loc[0, PROBABILITY_BD], places=5)

    def test_model_selection_SUMSTATS_BDEI(self):
        df_BD_vs_BDEI = modeldeep(BDEI_NWK, proba_sampling=P, vector_representation=SUMSTATS)
        print(df_BD_vs_BDEI)
        self.assertAlmostEqual(0.34999, df_BD_vs_BDEI.loc[0, PROBABILITY_BDEI], places=5)
        self.assertAlmostEqual(0.65001, df_BD_vs_BDEI.loc[0, PROBABILITY_BD], places=5)


    def test_estimate_bdei_FULL(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=FULL, model=BDEI)
        print(df)
        self.assertAlmostEqual(2.596412, df.loc[0, R], places=5)
        self.assertAlmostEqual(0.664282, df.loc[0, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(1.161259, df.loc[0, INCUBATION_PERIOD], places=5)

    def test_estimate_bdei_SUMSTATS(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDEI)
        print(df)
        self.assertAlmostEqual(2.416912, df.loc[0, R], places=5)
        self.assertAlmostEqual(0.724182, df.loc[0, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(1.009205, df.loc[0, INCUBATION_PERIOD], places=5)

    def test_estimate_bd_FULL(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=FULL, model=BD)
        print(df)
        self.assertAlmostEqual(2.315423, df.loc[0, R], places=5)
        self.assertAlmostEqual(1.069692, df.loc[0, INFECTIOUS_PERIOD], places=5)

    def test_estimate_bd_SUMSTATS(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BD)
        print(df)
        self.assertAlmostEqual(2.369476, df.loc[0, R], places=5)
        self.assertAlmostEqual(1.104765, df.loc[0, INFECTIOUS_PERIOD], places=5)


    def test_estimate_bdei_FULL_CI(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=FULL, model=BDEI, ci_computation=True)
        print(df)
        self.assertAlmostEqual(1.853841, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.814401, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(0.495808, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(0.978976, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(0.631817, df.loc[CI_lower, INCUBATION_PERIOD], places=5)
        self.assertAlmostEqual(1.796360, df.loc[CI_upper, INCUBATION_PERIOD], places=5)

    def test_estimate_bdei_SUMSTATS_CI(self):
        df = paramdeep(BDEI_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDEI, ci_computation=True)
        print(df)
        self.assertAlmostEqual(1.684386, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.594888, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(0.525116, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(1.058324, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(0.587087, df.loc[CI_lower, INCUBATION_PERIOD], places=5)
        self.assertAlmostEqual(1.654232, df.loc[CI_upper, INCUBATION_PERIOD], places=5)

    def test_estimate_bd_FULL_CI(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=FULL, model=BD, ci_computation=True)
        print(df)
        self.assertAlmostEqual(1.777894, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.150718, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(0.829689, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(1.456271, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)

    def test_estimate_bd_SUMSTATS_CI(self):
        df = paramdeep(BD_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BD, ci_computation=True)
        print(df)
        self.assertAlmostEqual(1.819216, df.loc[CI_lower, R], places=5)
        self.assertAlmostEqual(3.331554, df.loc[CI_upper, R], places=5)
        self.assertAlmostEqual(0.835348, df.loc[CI_lower, INFECTIOUS_PERIOD], places=5)
        self.assertAlmostEqual(1.493054, df.loc[CI_upper, INFECTIOUS_PERIOD], places=5)

    def test_chekdeep_bd_runs(self):
        png = 'BD_a_priori_check_BD_small.png'
        os.remove(png) if os.path.exists(png) else None
        checkdeep(BD_NWK, model=BD, outputfile_png=png)
        self.assertTrue(os.path.exists(png))
        os.remove(png) if os.path.exists(png) else None

    def test_chekdeep_bdei_runs(self):
        png = 'BDEI_a_priori_check_BDEI_small.png'
        os.remove(png) if os.path.exists(png) else None
        checkdeep(BDEI_NWK, model=BDEI, outputfile_png=png)
        self.assertTrue(os.path.exists(png))
        os.remove(png) if os.path.exists(png) else None
