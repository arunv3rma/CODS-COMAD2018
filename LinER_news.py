# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

from os import getcwd


# ############## Algorithm for estimating classifiers #################
def linear_estimated_classifiers(contexts, classifiers, costs, algorithm_parameters):
    # Algorithm parameters
    # alpha = algorithm_parameters[0]       # Scaling Parameter
    lambda_val = algorithm_parameters[1]    # Perturbation amount
    K = algorithm_parameters[2]             # Number od Arms/Classifiers
    d = algorithm_parameters[3]             # Dimension of Context
    m = algorithm_parameters[4]             # Exploration Steps
    R = algorithm_parameters[5]             # Variance of Noise
    delta = algorithm_parameters[6]         # Probability of mis-classification
    S = np.sqrt(algorithm_parameters[7])    # Maximum norm of Theta^*
    L = np.sqrt(algorithm_parameters[8])    # Maximum value of Square of norm of Context

    mu, sigma = 0, np.sqrt(R)               # Mean and Standard Deviation of Noise

    # Initialization of Data storage
    V = np.zeros((K, d, d))             # Initialization of data matrix
    V_inv = np.zeros((K, d, d))         # Initialization of inverse of data matrix
    M = lambda_val * np.zeros((K, d))   # Initial aggregated margin*context value
    Q = lambda_val * np.zeros((K, d))   # Initial estimate of Classifiers
    arm_counter = m * np.ones(K)        # Keeps how many time arm is used
    regret = []                         # Keeps Regret of each round

    for a in range(K):
        V[a] = lambda_val * np.identity(d)

    # Initial Testing Part
    for t in range(m):
        x_t = contexts[t][:]
        I_t = K
        x_t_outer = np.outer(x_t, x_t)
        for a in range(I_t):
            V[a] += x_t_outer
            margin = np.inner(classifiers[a], x_t)
            M[a] += margin * x_t + np.random.normal(mu, sigma, 1)[0]

    # Predicting classifiers based on sufficient field testing results
    V_inv[0] = np.linalg.inv(V[0])  # As same data for all Classifiers
    for a in range(K):
        V_inv[a] = V_inv[0]
        Q[a] = V_inv[0].dot(M[a])

    # Main part: Explore and Exploit Step
    Y = np.zeros(K)         # True label
    Y_cap = np.zeros(K)     # Estimated label
    I_t = 1                 # Best arm selected by algorithm
    optimal_arm = 1         # Optimal arm

    for t in range(m, data_points):
        x_t = contexts[t][:]

        # Estimate of labels by classifiers
        for a in range(K):
            Y[a] = np.inner(classifiers[a], x_t)

            beta_t = np.sqrt(lambda_val) * S + R * np.sqrt(2 * np.log(1 / delta) +
                                                           d * np.log(1 + ((arm_counter[a] * L) / (lambda_val * d))))

            conf_half_width = beta_t * x_t.dot(V_inv[a]).dot(x_t)
            Y_cap[a] = np.inner(Q[a], x_t) + conf_half_width

        # Finding best arm to use by algorithm
        for a in range(K):
            best_arm_flag = 1
            for b in range(a + 1, K):  # ***Jump directly to mismatch arm***
                if Y_cap[a] * Y_cap[b] < 0:
                    best_arm_flag = 0
                    break

            if best_arm_flag == 1:
                I_t = a + 1
                break

        # Optimal arm using all classifiers
        for a in range(K):
            best_arm_flag = 1
            for b in range(a + 1, K):  # ***Jump directly to mismatch arm***
                if Y[a] * Y[b] < 0:
                    best_arm_flag = 0
                    break

            if best_arm_flag == 1:
                optimal_arm = a + 1
                break

        # Updating Arm selection details
        if optimal_arm == I_t:
            regret.append(0)
        else:
            r_t = costs[I_t - 1] - costs[optimal_arm - 1] + 1  # Keep this '1' for mistake regret
            regret.append(r_t)

        # Update Historical data
        for a in range(I_t):
            arm_counter[a] += 1
            V[a] += np.outer(x_t, x_t)
            V_inv[a] = np.linalg.inv(V[a])
            M[a] += Y[a] * x_t + np.random.normal(mu, sigma, 1)[0]
            Q[a] = V_inv[a].dot(M[a])

    return regret


# Mistake based regret
def accumulative_mistake_regret(instantaneous_regret):
    time_horizon = [0]
    count = 0
    dis_count = 0
    batch = data_points / 10
    mismatch = [0]
    for s in range(len(instantaneous_regret)):
        count += 1
        if instantaneous_regret[s] == 1:
            dis_count += 1
        if count == batch:
            count = 0
            time_horizon.append(time_horizon[-1] + batch)
            mismatch.append(dis_count)

    if len(instantaneous_regret) % (1.0 * batch) != 0:
        time_horizon.append(time_horizon[-1] + count)
        mismatch.append(dis_count)

    return time_horizon, mismatch


# Mistake based regret
def accumulative_cost_regret(instantaneous_regret):
    time_horizon = [0]
    count = 0
    batch_regret = 0
    batch = data_points / 10
    mismatch = [0]
    for s in range(len(instantaneous_regret)):
        count += 1
        batch_regret += instantaneous_regret[s]
        if count == batch:
            count = 0
            time_horizon.append(time_horizon[-1] + batch)
            mismatch.append(batch_regret)

    if len(instantaneous_regret) % (1.0 * batch) != 0:
        time_horizon.append(time_horizon[-1] + count)
        mismatch.append(batch_regret)

    return time_horizon, mismatch


# Getting Average regret and Confidence interval
def accumulative_regret_error(regret):
    time_horizon = [0]
    samples = len(regret[0])
    runs = len(regret)
    batch = samples / 10

    # Time horizon
    t = 0
    while True:
        t += 1
        if time_horizon[-1] + batch > samples:
            if time_horizon[-1] != samples:
                time_horizon.append(time_horizon[-1] + samples % batch)
            break
        time_horizon.append(time_horizon[-1] + batch)

    # Mean batch regret of R runs
    avg_batched_regret = []
    for r in range(runs):
        count = 0
        accumulative_regret = 0
        batch_regret = [0]
        for s in range(samples):
            count += 1
            accumulative_regret += regret[r][s]
            if count == batch:
                batch_regret.append(accumulative_regret)
                count = 0

        if samples % batch != 0:
            batch_regret.append(accumulative_regret)
        avg_batched_regret.append(batch_regret)

    regret = np.mean(avg_batched_regret, axis=0)

    # Confidence interval
    conf_regret = []
    freedom_degree = runs - 2
    for r in range(len(avg_batched_regret[0])):
        conf_regret.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(np.array(avg_batched_regret)[:, r]))
    return time_horizon, regret, conf_regret


# Regret Plotting
def regret_plotting(regret, classifier_count):
    colors = list("rgbcmyk")
    shape = ['--*', '--+', '--H', '--d', '--X', '--^', '--v']
    # Scatter Error bar with scatter plot
    for a in range(len(classifier_count)):
        horizon, batched_regret, error = accumulative_regret_error(np.array(regret[a]))
        plt.errorbar(horizon, batched_regret, error, color=colors[a])
        plt.plot(horizon, batched_regret, colors[a] + shape[a], label=str(classifier_count[a]))

    # Location of the legend
    plt.legend(loc='upper left', numpoints=1)
    plt.title("Regret of Estimated Classifier based Algorithm for USS_Contextual")
    plt.ylabel("Expected Cumulative Regret")
    plt.xlabel("Horizon")
    plt.savefig("output/plots/regret_news50_" + str(data_points) + ".png", bbox_inches='tight')
    plt.close()


# ####################### Reading given data #########################
# Classifiers and costs
classifiers_list = [
    [0.00689261170327, 0.000118521417855, -0.00398044193028, -0.00610956668818, -0.00631240716371, 0.145796758359,
     -0.0369138884827, -0.021460026594, 0.00805765112636, 0.0160771599919, 0.0250421927904, 0.000392833800293,
     0.00667064660657, -0.0605982871105, 0.000153621166232, 0.0423370954143, 0.00529536964024, -0.000198502837787,
     6.69510233957e-07, 9.94277654042e-05, 3.84734270081e-06, 3.40456982621e-06, -6.05725810363e-06, -1.21780446191e-05,
     -2.12592756626e-05, 8.06708191817e-05, -3.62402338495e-05, -0.000200937814612, -1.68522565242e-05,
     -0.00613228716662, -1.66798044063e-05, 1.40246960281e-05, -2.02806010786e-05, 1.38638845307e-05, 1.26097387066e-05,
     7.91249950048e-06, 2.05222382071e-05, -0.000188685469277, -0.00844420286192, -0.00911484286249, -0.00806071642766,
     0.0196875999474, 0.00490844257139, 0.00216445047291, -4.04965626066e-05, 0.000109640736324, -0.0111746594166,
     0.0050537177794, 0.0125513502457, 0.0115077121997, 0.018954305693, -0.00445242031822, -0.00592993041954,
     -0.00307060656741, 0.0227462942707, -0.00837976063184, -0.0106075669523, 0.00759461560976, -0.00612083675334],
    [0.00923256029562, -0.00597288978005, -0.00396070960993, -0.00611583682309, -0.00625944926231, 0.148542337005,
     -0.0395713591969, -0.0218178176914, 0.00798607522519, 0.0176440619228, 0.0274645903861, 9.15173024632e-05,
     0.0067054896844, -0.0628263838879, 0.000156649212949, 0.0445597899861, 0.00544566388011, -0.000128271611434,
     -3.85946630414e-05, 0.000150816787372, -4.70817637664e-06, 1.24183529313e-05, -1.83567118252e-05,
     6.17206534893e-05, -5.03878055964e-05, 6.42799375202e-05, -0.000119711981499, 0.000106884522403,
     -0.000133524620379, -0.00613845042114, -1.65170195826e-05, 1.36042365971e-05, -2.00752659886e-05,
     1.39459552638e-05, 1.26481286629e-05, 7.78464119617e-06, 2.04327698591e-05, 5.06741281161e-05, -0.00862043380913,
     -0.00943850379564, -0.00842744992405, 0.0203086429045, 0.00517290116252, 0.0023122428346, -2.6021256857e-05,
     0.000134348666431, -0.0115386524816, 0.00541161690464, 0.0131767441828, 0.0120686911425, 0.0198972632592,
     -0.00422474437518, -0.00571259145729, -0.002814658875, 0.0232721847559, -0.00834772076934, -0.0108919660091,
     0.00774449299531, -0.006127059745],
    [-0.000925541388136, 0.000883854889994, -0.00282815950823, -0.00421618924478, -0.00512043146168, 0.117938941737,
     -0.010724153364, -0.00631115135156, 0.00688836693607, 0.00621643125444, 0.00200293796763, 0.00331808490741,
     0.00572414396112, -0.0345003165227, 0.000122437354273, 0.0180310927125, 0.00426555040803, -0.000595153037587,
     -3.42698657475e-05, -2.60488485419e-05, 4.66084276041e-06, 7.62253681822e-06, 9.20627786841e-06,
     -3.94022019707e-06, -0.000134494994748, 1.05200312138e-05, 0.000186414492227, -0.00017783118056, 0.000287757459866,
     -0.00423444571588, -1.7363427467e-05, 1.54972456121e-05, -2.05829974399e-05, 1.29547218529e-05, 1.28349848032e-05,
     8.17194572011e-06, 2.10069305233e-05, -0.00176053391439, -0.00666805466174, -0.00469181623505, -0.00353136755015,
     0.0124288310983, 0.00326961272149, 0.00132350595921, -0.00012589848139, -0.000151022643, -0.00559411150777,
     0.00136956724686, 0.00716957099242, 0.00619942208909, 0.010807881041, -0.00694322429361, -0.00798272243783,
     -0.00581034996128, 0.0153728689214, -0.00816413409921, -0.00582476869641, 0.00505054981395, -0.0042229332428],
    [-0.00837183598266, -0.000154513607367, -0.00084901594275, -0.00119160981738, -0.0012789437537, 0.0209009816908,
     0.00134326592401, -0.00184211054094, 0.000833619983263, -0.00111009892979, -0.0125966365341, 0.00048950035086,
     0.00181662223674, -0.00397611005022, 0.000121416621813, -0.00013111091535, 0.00199468165964, -0.000699133458027,
     1.44285440435e-06, 4.86628217279e-05, -9.6477922268e-06, -5.75940304787e-06, 9.17628920046e-06, 5.68597912204e-06,
     8.52708559123e-05, 0.000100923118389, -0.000133759971347, 2.91107115863e-05, 3.35065251243e-05, -0.0012078126005,
     -1.83961537918e-05, 1.82921687481e-05, -1.92589960695e-05, 8.75692737967e-06, 1.56575104293e-05, 9.78587652702e-06,
     2.54433869564e-05, -0.000343664199678, -0.00187510206132, 2.68699601344e-05, -0.000288277591129, 0.00128719448961,
     0.000411569011514, -6.27844168231e-05, -8.55214386741e-05, -5.02199488832e-05, -0.00117142992735,
     -2.44878541457e-05, 0.00081157952039, 0.000772429110547, 0.000817569843015, -0.0017938443306, -0.00223757871044,
     -0.00140016867736, 0.00113600522586, -0.00341012930085, -0.000708241672943, -0.000825540290183, -0.00119297526728],
    [0.00170837514241, -0.00447840516909, -0.00353691840407, -0.00535091472366, -0.00597680099554, 0.135095428213,
     -0.023761120675, -0.0105907461093, 0.00794776404287, 0.0107705971693, 0.0146485477421, 0.00299664003875,
     0.0063988182009, -0.0486381782106, 0.000136695708934, 0.0297058178481, 0.00491433649822, -0.000305027683801,
     2.95921953019e-05, 0.000143963959309, -7.16069763622e-07, 3.21082433333e-06, -1.9232532687e-05, 8.4984032937e-06,
     -9.65712288295e-06, 0.000152325514356, 5.50880545392e-06, 0.000171129633307, 0.000121762879256, -0.0053724017039,
     -1.68693835452e-05, 1.4942969081e-05, -2.08322791106e-05, 1.41755748247e-05, 1.26134185971e-05, 7.95403957458e-06,
     2.05674581717e-05, -0.00155121177964, -0.00783127871472, -0.00704747698348, -0.00595026701865, 0.0170198074375,
     0.00413683086237, 0.0017275740705, -9.00369408148e-05, -1.79429941725e-05, -0.00866917901377, 0.00330816640822,
     0.00986598974249, 0.00881086876084, 0.0149278820422, -0.00609372529547, -0.0075674412302, -0.00466943329822,
     0.0195314261827, -0.00870681877975, -0.00834557411, 0.00646038103578, -0.00536041736447]
    ]

classifiers2 = [classifiers_list[0], classifiers_list[2]]
classifiers3 = [classifiers_list[0], classifiers_list[2], classifiers_list[4]]
classifiers4 = [classifiers_list[0], classifiers_list[1], classifiers_list[3], classifiers_list[4]]

synthetic_classifiers = np.array([classifiers2, classifiers3, classifiers4, classifiers_list])

classifier_costs = [[0.2, 0.45],
                    [0.2, 0.45, 0.95],
                    [0.2, 0.30, 0.68, 0.95],
                    [0.2, 0.30, 0.45, 0.68, 0.95]
                    ]
#
# synthetic_classifiers = np.array([classifiers])
# classifier_costs = [[0.12, 0.28, 0.45, 0.68, 0.95]]

# Algorithm parameters: [alpha, lambda_val, K, d, m, R, delta, S, L]
parameters = [1, 0.5, 2, 59, 1, 0.01, 0.1, 0.034, 2.9620438465e+12]

classifiers_count = []
for c in range(len(synthetic_classifiers)):
    classifiers_count.append(len(synthetic_classifiers[c]))

# Reading Synthetically generated Contextual Data
path_to_data = getcwd() + "/input/realDataset/processed/"
fileName = path_to_data + "news.csv"
data_samples = pd.read_csv(fileName, header=None).iloc[:, 0:-1]
data_samples[len(data_samples.columns)] = 1
data_points = len(data_samples)

parameters[3] = len(data_samples.columns)
classifier_regret = []
for c in range(len(synthetic_classifiers)):
    file_name = open("output/regretFiles/news/" + str(data_points) + "_" + str(classifiers_count[c]) + ".txt", 'w')
    parameters[2] = classifiers_count[c]
    run_regret = []
    for i in range(1, 21):
        data = data_samples.iloc[:, :].sample(frac=1).as_matrix()
        iter_regret = linear_estimated_classifiers(data, synthetic_classifiers[c], classifier_costs[c], parameters)
        run_regret.append(iter_regret)
        file_name.write(str(iter_regret) + "\n")
    classifier_regret.append(run_regret)
    file_name.close()

regret_plotting(classifier_regret, classifiers_count)
