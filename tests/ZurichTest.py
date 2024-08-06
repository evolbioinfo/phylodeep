import os
import unittest

from paramdeep import paramdeep
from phylodeep import FULL, BDSS, SUMSTATS, BDEI, BD

from modeldeep import modeldeep

CI_upper = 'ci_97_5_boundary'

CI_lower = 'ci_2_5_boundary'

P = 0.25

ZURICH_NWK = os.path.join(os.path.dirname(__file__), '..', 'test_tree_HIV_Zurich', 'Zurich.trees')


class ZurichTest(unittest.TestCase):

    def test_model_selection_FULL(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL)
        self.assertAlmostEqual(0.000005, df_BDEI_vs_BD_vs_BDSS.loc[0, 'Probability_BDEI'], places=6)
        self.assertAlmostEqual(0.999995, df_BDEI_vs_BD_vs_BDSS.loc[0, 'Probability_BDSS'], places=6)
        self.assertAlmostEqual(0, df_BDEI_vs_BD_vs_BDSS.loc[0, 'Probability_BD'], places=6)

    def test_model_selection_SUMSTATS(self):
        df_BDEI_vs_BD_vs_BDSS = modeldeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS)
        self.assertAlmostEqual(0.000005, df_BDEI_vs_BD_vs_BDSS.loc[0, 'Probability_BDEI'], places=6)
        self.assertAlmostEqual(0.999995, df_BDEI_vs_BD_vs_BDSS.loc[0, 'Probability_BDSS'], places=6)
        self.assertAlmostEqual(0, df_BDEI_vs_BD_vs_BDSS.loc[0, 'Probability_BD'], places=6)

    def test_estimate_bdss_FULL(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL, model=BDSS)
        self.assertAlmostEqual(1.687666, df.loc[0, 'R_naught'], places=6)
        self.assertAlmostEqual(9.783522, df.loc[0, 'Infectious_period'], places=6)
        self.assertAlmostEqual(9.336288, df.loc[0, 'X_transmission'], places=6)
        self.assertAlmostEqual(0.078699, df.loc[0, 'Superspreading_individuals_fraction'], places=6)

    def test_estimate_bdss_SUMSTATS(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDSS)
        self.assertAlmostEqual(1.602880, df.loc[0, 'R_naught'], places=6)
        self.assertAlmostEqual(10.21694, df.loc[0, 'Infectious_period'], places=6)
        self.assertAlmostEqual(8.781717, df.loc[0, 'X_transmission'], places=6)
        self.assertAlmostEqual(0.071957, df.loc[0, 'Superspreading_individuals_fraction'], places=6)

    def test_estimate_bdei_FULL(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL, model=BDEI)
        self.assertAlmostEqual(1.755864, df.loc[0, 'R_naught'], places=6)
        self.assertAlmostEqual(2.370077, df.loc[0, 'Infectious_period'], places=6)
        self.assertAlmostEqual(5.632354, df.loc[0, 'Incubation_period'], places=6)

    def test_estimate_bdei_SUMSTATS(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDEI)
        self.assertAlmostEqual(1.4980435, df.loc[0, 'R_naught'], places=6)
        self.assertAlmostEqual(1.7075282, df.loc[0, 'Infectious_period'], places=6)
        self.assertAlmostEqual(5.7501006, df.loc[0, 'Incubation_period'], places=6)

    def test_estimate_bd_FULL(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL, model=BD)
        self.assertAlmostEqual(1.533808, df.loc[0, 'R_naught'], places=6)
        self.assertAlmostEqual(8.028381, df.loc[0, 'Infectious_period'], places=6)

    def test_estimate_bd_SUMSTATS(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BD)
        self.assertAlmostEqual(1.466705, df.loc[0, 'R_naught'], places=6)
        self.assertAlmostEqual(8.684904, df.loc[0, 'Infectious_period'], places=6)

    def test_estimate_bdss_FULL_CI(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL, model=BDSS, ci_computation=True)
        self.assertAlmostEqual(1.402187, df.loc[CI_lower, 'R_naught'], places=6)
        self.assertAlmostEqual(2.076712, df.loc[CI_upper, 'R_naught'], places=6)
        self.assertAlmostEqual(8.120945, df.loc[CI_lower, 'Infectious_period'], places=6)
        self.assertAlmostEqual(12.260722, df.loc[CI_upper, 'Infectious_period'], places=6)
        self.assertAlmostEqual(6.651802, df.loc[CI_lower, 'X_transmission'], places=6)
        self.assertAlmostEqual(10, df.loc[CI_upper, 'X_transmission'], places=6)
        self.assertAlmostEqual(0.050490, df.loc[CI_lower, 'Superspreading_individuals_fraction'], places=6)
        self.assertAlmostEqual(0.132575, df.loc[CI_upper, 'Superspreading_individuals_fraction'], places=6)

    def test_estimate_bdss_SUMSTATS_CI(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDSS, ci_computation=True)
        self.assertAlmostEqual(1.345565, df.loc[CI_lower, 'R_naught'], places=6)
        self.assertAlmostEqual(1.960975, df.loc[CI_upper, 'R_naught'], places=6)
        self.assertAlmostEqual(8.328682, df.loc[CI_lower, 'Infectious_period'], places=6)
        self.assertAlmostEqual(12.811870, df.loc[CI_upper, 'Infectious_period'], places=6)
        self.assertAlmostEqual(5.921604, df.loc[CI_lower, 'X_transmission'], places=6)
        self.assertAlmostEqual(10, df.loc[CI_upper, 'X_transmission'], places=6)
        self.assertAlmostEqual(0.05, df.loc[CI_lower, 'Superspreading_individuals_fraction'], places=6)
        self.assertAlmostEqual(0.125339, df.loc[CI_upper, 'Superspreading_individuals_fraction'], places=6)

    def test_estimate_bdei_FULL_CI(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL, model=BDEI, ci_computation=True)
        self.assertAlmostEqual(1.511312, df.loc[CI_lower, 'R_naught'], places=6)
        self.assertAlmostEqual(2.151845, df.loc[CI_upper, 'R_naught'], places=6)
        self.assertAlmostEqual(1.816947, df.loc[CI_lower, 'Infectious_period'], places=6)
        self.assertAlmostEqual(3.039504, df.loc[CI_upper, 'Infectious_period'], places=6)
        self.assertAlmostEqual(3.715558, df.loc[CI_lower, 'Incubation_period'], places=6)
        self.assertAlmostEqual(7.990896, df.loc[CI_upper, 'Incubation_period'], places=6)

    def test_estimate_bdei_SUMSTATS_CI(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BDEI, ci_computation=True)
        self.assertAlmostEqual(1.302846, df.loc[CI_lower, 'R_naught'], places=6)
        self.assertAlmostEqual(1.778539, df.loc[CI_upper, 'R_naught'], places=6)
        self.assertAlmostEqual(1.269457, df.loc[CI_lower, 'Infectious_period'], places=6)
        self.assertAlmostEqual(2.362872, df.loc[CI_upper, 'Infectious_period'], places=6)
        self.assertAlmostEqual(3.862964, df.loc[CI_lower, 'Incubation_period'], places=6)
        self.assertAlmostEqual(7.991413, df.loc[CI_upper, 'Incubation_period'], places=6)

    def test_estimate_bd_FULL_CI(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=FULL, model=BD, ci_computation=True)
        self.assertAlmostEqual(1.338377, df.loc[CI_lower, 'R_naught'], places=6)
        self.assertAlmostEqual(1.812531, df.loc[CI_upper, 'R_naught'], places=6)
        self.assertAlmostEqual(6.874818, df.loc[CI_lower, 'Infectious_period'], places=6)
        self.assertAlmostEqual(9.803920, df.loc[CI_upper, 'Infectious_period'], places=6)

    def test_estimate_bd_SUMSTATS_CI(self):
        df = paramdeep(ZURICH_NWK, proba_sampling=P, vector_representation=SUMSTATS, model=BD, ci_computation=True)
        self.assertAlmostEqual(1.282737, df.loc[CI_lower, 'R_naught'], places=6)
        self.assertAlmostEqual(1.709024, df.loc[CI_upper, 'R_naught'], places=6)
        self.assertAlmostEqual(7.382627, df.loc[CI_lower, 'Infectious_period'], places=6)
        self.assertAlmostEqual(10.686424, df.loc[CI_upper, 'Infectious_period'], places=6)
