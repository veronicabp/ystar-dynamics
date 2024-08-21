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
gen ystar=_b[/ystar]
gen ub= _b[/ystar] + 1.96*_se[/ystar]
gen lb= _b[/ystar] - 1.96*_se[/ystar]

keep xaxis ystar ub lb 
gen xaxis12 = xaxis*12
drop if missing(xaxis)
tempfile pre2003
save `pre2003'

use "$clean/ystar_yearly.dta", clear
gen yearly = 1
keep if year<=2022
append using "$clean/ystar_monthly.dta"
drop if xaxis12==2003*12

keep if yearly==1 | xaxis12 >= 12*$year1
drop if ystar==.

gen xaxis = xaxis12/12
sort xaxis
keep ystar ub lb xaxis xaxis12
append using `pre2003'

gsort xaxis12

local x = $year1 + 2.4
local y = ystar[_N] - 0.25

// Without forward rate
twoway 	(line ystar xaxis if xaxis>=2003, lcolor(black) lpattern(solid)) ///
		(line ystar xaxis if xaxis<=2003, lcolor(black) lpattern(dash)) ///
		(rarea ub lb xaxis if xaxis12 < 12*$year1 + $month1 - 1, color(gs10%30) lcolor(%0)) ///
		(scatter ystar xaxis if xaxis12 >= 12*$year1 - 1, mcolor("$accent1") msymbol(O) msize(vsmall)) ///
		(scatter ystar xaxis if xaxis12 == 12*$year1 + $month1 - 1, mcolor("$accent1") msymbol(O) msize(vsmall)), ///
		legend(order(4 "Monthly Real-Time Estimates") ring(0) position(2)) ///
		xtitle("") ytitle("y*") xlabel(2000(5)2026.5) ylabel(2(1)6) ///
		text(`y' `x' "$month1_str" "$year1", color($accent1))
graph export "$fig/ystar_timeseries.png", replace

// Export ystar timeseries 
gen month = round((xaxis - int(xaxis))*12) + 1 if xaxis>=2023
gen year = int(xaxis)
gsort year month

tostring year, replace 
replace year = "2000-2003" if _n<=4

gen se = (ub-ystar)/1.96

keep year month  ystar se 
order year month  ystar se 

gduplicates drop
export delimited "$clean/ystar_estimates.csv", replace
