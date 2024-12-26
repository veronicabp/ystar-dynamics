**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*************************************
* Figure 1: Dynamics of y*
*************************************

use "$clean_update/ystar_estimates.dta", clear
rename date xaxis

local x = $year1 + 2.4
local y = ystar[_N] - 0.25
local last_date = xaxis[_N]

// Without forward rate
twoway 	(line ystar xaxis if xaxis>=2003, lcolor(black) lpattern(solid)) ///
		(line ystar xaxis if xaxis<=2003, lcolor(black) lpattern(dash)) ///
		(rarea ub lb xaxis if xaxis != `last_date', color(gs10%30) lcolor(%0)) ///
		(scatter ystar xaxis if xaxis >= $year1, mcolor("$accent1") msymbol(O) msize(vsmall)), ///
		legend(order(4 "Monthly Real-Time Estimates") ring(0) position(2)) ///
		xtitle("") ytitle("y*") xlabel(2000(5)2026.5) ylabel(2(1)6) ///
		text(`y' `x' "$month1_str" "$year1", color($accent1))
graph export "$fig/ystar_timeseries.png", replace
