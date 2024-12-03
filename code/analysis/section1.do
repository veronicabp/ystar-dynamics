**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*************************************
* Figure 1: Dynamics of y*
*************************************

// For pre-2003, get mean y* for all years
use "$clean/experiments.dta", clear
gen xaxis = _n+1999 if _n<=4
nl $nlfunc if year<=2003, initial(ystar 3) variables(did_rsi T k) vce(robust)
gen ystar_pre2003=_b[/ystar]
gen se_pre2003= _se[/ystar]

keep xaxis ystar_pre2003 se_pre2003
drop if missing(xaxis)
tempfile pre2003
save `pre2003'

use "$clean/ystar_estimates.dta", clear
gen xaxis = year if year<= 2022
replace xaxis = year + (month-1)/12 if year>2022
append using `pre2003'

gen ystar = ystar_pre2003 if xaxis<=2003
replace ystar = ystar_yearly if xaxis>2003
replace ystar = ystar_monthly if xaxis>2023

gen se = se_pre2003 if xaxis<=2003
replace se = se_yearly if xaxis>2003
replace se = se_monthly if xaxis>2023

gen ub = ystar + 1.96*se
gen lb = ystar - 1.96*se 

keep xaxis ystar ub lb 
drop if missing(ystar)
duplicates drop
gsort xaxis 

local x = $year1 + 2.4
local y = ystar[_N] - 0.25
local last_date = xaxis[_N]

// Without forward rate
twoway 	(line ystar xaxis if xaxis>=2003, lcolor(black) lpattern(solid)) ///
		(line ystar xaxis if xaxis<=2003, lcolor(black) lpattern(dash)) ///
		(rarea ub lb xaxis if xaxis != `last_date', color(gs10%30) lcolor(%0)) ///
		(scatter ystar xaxis if xaxis >= $year1, mcolor("$accent1") msymbol(O) msize(vsmall)) ///
		(scatter ystar xaxis if xaxis == `last_date', mcolor("$accent1") msymbol(O) msize(vsmall)), ///
		legend(order(4 "Monthly Real-Time Estimates") ring(0) position(2)) ///
		xtitle("") ytitle("y*") xlabel(2000(5)2026.5) ylabel(2(1)6) ///
		text(`y' `x' "$month1_str" "$year1", color($accent1))
graph export "$fig/ystar_timeseries.png", replace
