**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

// Save supply elasticity variables
use "$clean/hilber_lad21.dta", clear
keep lpa_code lpa_name pdevel90_m refusal_maj_7908 delchange_maj* 
gduplicates drop lpa_code, force 
tempfile hilber_time_invariant 
save `hilber_time_invariant'

* Calculate local housing beta
use "$clean/expanded_hilber_data.dta", clear
gegen lpa_num=group(lpa_code)
xtset lpa_num year
gen d_log_hpi = log_hpi - L.log_hpi 

local dep_var log_hpi
local indep_var log_earn

gen beta=.
reghdfe `dep_var' i.lpa_num#c.`indep_var', absorb(lpa_num year)
qui:levelsof lpa_num, local(lpas)
foreach lpa of local lpas {
	qui: replace beta = _b[`lpa'.lpa_num#c.`indep_var'] if lpa_num==`lpa'
}

gcollapse beta d_log_hpi, by(lpa_code)
save "$working/local_housing_beta.dta", replace

** Get pre and post ystar for each local authority
use "$clean/experiments.dta", clear 
local lhs did_rsi_yearly // Use yearly index to get more data

keep if strpos(lpa_code, "E")==1 // We only have elasticity data for England
drop if missing(`lhs')

local pre = 2009
local post = 2022

gen pre = year<=`pre'
gen post = year>=`post' 
gen all = 1

local tags "pre" "post" "all"

foreach tag in "`tags'" {
	gen ystar_`tag' = .
	gen var_`tag' = .
	gegen N_`tag' = total(`tag'), by(lpa_code)
}

levelsof lpa_code, local(lpas)
foreach lpa of local lpas {
	di "`lpa'"
	foreach tag in "`tags'" {
		qui: sum `lhs' if `tag' & lpa_code=="`lpa'"
		if r(N)<=5 continue
		cap nl (`lhs' = ln(1-exp(-({ystar=3}/100)*(T+k))) - ln(1-exp(-({ystar=3}/100)*T))) if `tag' & lpa_code=="`lpa'"
		qui: replace ystar_`tag' = _b[/ystar] if lpa_code=="`lpa'" & _se[/ystar]!=. & _se[/ystar]!=0
		qui: replace var_`tag' = (_se[/ystar])^2 if lpa_code=="`lpa'" & _se[/ystar]!=. & _se[/ystar]!=0
	}
}

collapse ystar_* var_* N_* (first) region, by(lpa_code)
gen d_ystar = ystar_post - ystar_pre

gen w = 1/(var_pre + var_post)
gen w_all = 1/var_all
cap winsor d_ystar, p(0.05) gen(d_ystar_win)

merge 1:1 lpa_code using `hilber_time_invariant', keep(match) nogen
merge 1:1 lpa_code using "$working/local_housing_beta.dta", keep(match) nogen

save "$clean/ystar_by_lpas_`pre'-`post'.dta", replace
