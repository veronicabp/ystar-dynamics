import sys
sys.path.append('')
from utils import *

from scipy.optimize import least_squares
import scipy.stats as stats

def significance_symbol(coeff, se):
    """Return significance symbols based on p-value."""
    
    t_stat = coeff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    if p_value < 0.001:
        return '\sym{{*}{*}{*}}'
    elif p_value < 0.01:
        return '\sym{{*}{*}}'
    elif p_value < 0.05:
        return '\sym{*}'
    else:
        return ''

def bootstrap_ses(df, n_bootstrap=1000):
    # Store the estimated parameters
    params_bootstrapped = np.zeros((n_bootstrap, 3))
    
    # Perform bootstrap resampling
    for i in tqdm(range(n_bootstrap)):
        # Resample the data with replacement
        resample_index = np.random.choice(df.index, size=len(df), replace=True)
        df_resampled = df.loc[resample_index]
        
        # Estimate parameters on the resampled data
        res = estimate_ystar_alpha(df_resampled)
        
        # Save the estimated parameters
        params_bootstrapped[i, :] = res.x

    # Compute the standard error for each parameter
    standard_errors = params_bootstrapped.std(axis=0)

    return standard_errors

def estimate_ystar_alpha(df):
    
    def model(params):
        ystar=params[0]/100
        alpha_u80=params[1]
        alpha_o80=params[2]
        
        p1 = 1 - np.exp(-ystar * (df['T'] + df['k']))
        p0 = 1 - np.exp(-ystar * df['T'])
        p0_option_val = (df['over80'] * (df['Pi'] * (1 - alpha_o80) + (1 - df['Pi']) * (1 - alpha_u80))) * (np.exp(-ystar * df['T']) - np.exp(-ystar * (df['T'] + 90)))
        did_est = np.log(p1) - np.log(p0 + p0_option_val)
        return did_est 
    
    def nlls(params):
        return model(params) - df['did_rsi']
    
    # Estimate ystar as if there were full holdup
    res = least_squares(nlls, x0=[3, 1, 1], bounds=([0,0,0], [np.inf,1,1]), loss='linear')
    return res

def estimate(df):
    print("Estimating Coefficients")
    coeffs = estimate_ystar_alpha(df).x
    print("Getting Standard Errors")
    ses = bootstrap_ses(df)
    return coeffs, ses
    
file = os.path.join(clean_folder, 'experiments.dta')
df = pd.read_stata(file)
df = df[~df['did_rsi'].isna()]
df = df[df.year>=2003].copy()
df['over80'] = (df['T']>80)
pre = df[df.year<=2010]
post = df[df.year>2010]


coeffs0, ses0 = estimate(pre)
coeffs1, ses1 = estimate(post)

coefficients = {'pre':coeffs0, 'post':coeffs1}
std_errors = {'pre':ses0, 'post':ses1}

# Create table
num_pre = len(pre)
num_post = len(post)

# Assuming the coefficients, standard errors and p-values are structured like this:
variables = [r"$y^*$", r"$\alpha_{t}^{H}$", r"$\alpha_{t}^{L}$"]

# Create the LaTeX table
latex_table = r"\begin{tabular}{lcc}" + "\n"
latex_table += r"\hline" + "\n"
latex_table += r"& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)}\\" + "\n"
latex_table += r"\hline" + "\n"

for i, var in enumerate(variables):
    latex_table += f"{var}"
    for period in coefficients.keys():
        latex_table += f"& {coefficients[period][i]:.2f}{significance_symbol(coefficients[period][i], std_errors[period][i])}"
    latex_table += r"\\" + "\n"
    for period in coefficients.keys():
        latex_table += f"& ({std_errors[period][i]:.2f})"
    latex_table += r"\\" + "\n"

latex_table += r"\hline" + "\n"
latex_table += r"Period & Pre 2010 & Post 2010\\" + "\n"
latex_table += f"N & {num_pre:,} & {num_post:,}" + r"\\" + "\n"
latex_table += r"\hline" + "\n"
latex_table += r"\hline" + "\n"
latex_table += r"\multicolumn{2}{l}{{\footnotesize{}Standard errors in parentheses}}\\" + "\n"
latex_table += r"\multicolumn{2}{l}{{\footnotesize{}\sym{*} $p<0.05$, \sym{{*}{*}} $p<0.01$, \sym{{*}{*}{*}} $p<0.001$}}" + "\n"
latex_table += r"\end{tabular}"

file = os.path.join(tables_folder, 'estimate_alphas.tex')
with open(file, "w") as f:
    f.write(latex_table)


