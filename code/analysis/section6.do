**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*****************************************
* Figure 11: Comparison With Other Assets
*****************************************

****** Yearly regressions ******
use "$clean/uk_interest_rates.dta", clear
collapse uk*, by(year)

gen date = year
merge 1:1 date using "$clean/ystar_estimates.dta", nogen keep(master match)
drop date

merge 1:1 year using "$clean/global_forward.dta", nogen
merge 1:1 year using "$clean/global_rtp.dta", nogen
merge 1:1 year using "$clean/uk_rtp.dta", nogen

tset year
gen w=1/(se^2)

* Adjust interest rates assuming 3% inflation
gen rate10y20_real = rate10y20-3

* Normalize rent-to-price to 2023 level of y*
drop if year>2023
gsort -year 
gen rtp_uk = ystar if year==2023
replace rtp_uk = rtp_uk[_n-1] * rent_price_uk/rent_price_uk[_n-1] if _n>1

gen rtp_global = ystar if year==2023
replace rtp_global = rtp_uk[_n-1] * rent_price_global/rent_price_global[_n-1] if _n>1

gen xaxis = year

***** Plots
twoway 	(line ystar xaxis if xaxis>=2003, lcolor(black) lpattern(solid)) ///
		(line ystar xaxis if xaxis<=2003, lcolor(black) lpattern(dash)) ///
		(rarea ub lb xaxis, color(gs10%30) lcolor(%0)) ///
		(line rtp_uk xaxis, lpattern(dash) lcolor("$accent1")) ///
		(line rtp_global xaxis, lpattern(dash) lcolor("$accent2")) if year>=2000, ///
		legend(order(1 "y* (Left Axis)" ///
					 4 "UK Rent-to-Price Ratio" ///
					 5 "Global Rent-to-Price Ratio") ring(0) position(7)) ///
		xtitle("") ytitle("")  xlabel(2000(5)2023) ///
		ylabel(2(1)6, axis(1))
graph export "$fig/ystar_rtp_timeseries.png", replace

twoway 	(line ystar xaxis if xaxis>=2003, lcolor(black) lpattern(solid)) ///
		(line ystar xaxis if xaxis<=2003, lcolor(black) lpattern(dash)) ///
		(rarea ub lb xaxis, color(gs10%30) lcolor(%0)) ///
		(line uk10y15_real xaxis, yaxis(2) lpattern(dash) lcolor("$accent1")) ///
		(line rate10y20_real xaxis, yaxis(2) lpattern(dash) lcolor("$accent2")) if year>=2000, ///
		legend(order(1 "y* (Left Axis)" ///
					 4 "UK 10 Year 15 Real Forward Rate (Right Axis)" ///
					 5 "Global 10 Year 20 Real Forward Rate (Right Axis)") ring(0) position(7)) ///
		xtitle("") ytitle("") ytitle("", axis(2)) xlabel(2000(5)2023) ///
		ylabel(2(1)6, axis(1)) ylabel(-2(1)2 , axis(2))
graph export "$fig/ystar_forward_timeseries.png", replace

*****************************************
* Table 5: Supply elasticity
*****************************************

use "$clean/ystar_by_lpas_2009-2022.dta", clear
drop if missing(d_ystar)

gen refusal = refusal_maj_7908

eststo clear
eststo: reg d_ystar refusal [aw=w], vce(robust)
eststo: reg d_ystar refusal pdevel90_m2 [aw=w], vce(robust)

// Region FE
eststo: reg d_ystar refusal pdevel90_m2 [aw=w], vce(robust) absorb(region)
estadd local regionfe "\checkmark", replace

// Instrument reduced form
eststo: reg d_ystar delchange_maj5 [aw=w], vce(robust)

// First stage
eststo: reg refusal delchange_maj5, vce(robust)
// predict refusal_predicted
// replace refusal=refusal_predicted

// IV
// eststo: reg d_ystar refusal [aw=w], robust
// eststo: reg d_ystar pdevel90_m2 refusal [aw=w], robust
eststo: ivreg2 d_ystar (refusal=delchange_maj5) [aw=w], robust
eststo: ivreg2 d_ystar pdevel90_m2 (refusal=delchange_maj5) [aw=w], robust


esttab using "$tab/supply_elasticity.tex", ///
	mgroups("$\Delta y^*$" "Refusal Rate" "$\Delta y^*$", pattern(1 0 0 0 1 1) ///
				prefix(\multicolumn{@span}{c}{) suffix(}) ///
				span erepeat(\cmidrule(lr){@span})) ///
	mtitle("OLS" "OLS" "OLS" "OLS" "1st Stage" "IV" "IV") ///
	keep(refusal pdevel90_m2 delchange_maj5) ///
	varlabel(	refusal "Refusal Rate" ///
				pdevel90_m2 "Share Developed" ///
				delchange_maj5 "Change in Delay Rate") ///
	stats(regionfe N r2, label("Region FE" "N" "R2") fmt(0 0 2)) ///
	se(2) b(2) replace
	
*****************************************
* Figure 12: Supply elasticity
*****************************************

reg refusal_maj_7908 delchange_maj5 pdevel90_m2 [aw=w]
predict refusal_predicted

reghdfe refusal_predicted pdevel90_m2 [aw=w], noabsorb residuals(refusal_predicted_res)
sum refusal_predicted [aw=w]
replace refusal_predicted_res = refusal_predicted_res + r(mean)

reghdfe d_ystar pdevel90_m2 [aw=w], noabsorb residuals(d_ystar_res)
sum d_ystar [aw=w]
replace d_ystar_res = d_ystar_res + r(mean)

binscatter2 d_ystar_res refusal_predicted_res [aw=w], xtitle("Predicted Refusal Rate") ytitle("∆y*")
graph export "$fig/supply_elasticity_binscatter.png", replace

*****************************************
* Table 6: Risk Premium
*****************************************
use "$clean/ystar_by_lpas_2009-2022.dta", clear
drop if missing(d_ystar)

reg refusal_maj_7908 delchange_maj5
predict refusal_predicted

eststo clear
eststo: reg beta refusal_predicted, vce(robust)
eststo: reg ystar_all beta [aw=w_all], vce(robust)
eststo: reg ystar_all beta refusal_predicted [aw=w_all], vce(robust)
eststo: reg d_ystar beta [aw=w], vce(robust)
eststo: reg d_ystar beta refusal_predicted [aw=w], vce(robust)

esttab using "$tab/cross_sectional_risk.tex", ///
	b(2) se(2) keep(beta refusal_predicted) ///
	order(beta refusal_predicted) ///
	varlabels(beta "Housing $\beta$" refusal_predicted "Predicted Refusal Rate") ///
	mgroups("Housing $\beta$" "$ y^*$" "$\Delta y^*$", pattern(1 1 0 1) ///
				prefix(\multicolumn{@span}{c}{) suffix(}) ///
				span erepeat(\cmidrule(lr){@span})) ///
	nomtitle ///
	stats(N r2, label("N" "R2") fmt(0 2)) replace


*****************************************
* Figure 13: Risk Premium
*****************************************

binscatter2 ystar_all beta [aw=w], xtitle("Local Housing Beta") ytitle("y*")
graph export "$fig/cross_sectional_risk_binscatter.png", replace


*****************************************
* Table: Climate Risk
*****************************************
use "$clean/ystar_by_lpas_2009-2022.dta", clear
drop if missing(d_ystar)

eststo clear
eststo: reg ystar_all flood_risk_share  [aw=w_all], vce(robust)
eststo: reg ystar_all probable_risk_2030_share  [aw=w_all], vce(robust)

esttab using "$tab/climate_risk.tex", ///
	mgroups("$ y^*$", pattern(1 0) ///
				prefix(\multicolumn{@span}{c}{) suffix(}) ///
				span erepeat(\cmidrule(lr){@span})) ///
	nomtitle ///
	keep(flood_risk_share probable_risk_2030_share) ///
	varlabel(	flood_risk_share "Flood Risk" ///
				probable_risk_2030_share "Subsidence Risk") ///
	stats(N r2, label("N" "R2") fmt(0 2)) ///
	se(2) b(2) replace

// eststo clear
// eststo: reg ystar_all flood_risk_share  [aw=w_all], vce(robust)
// eststo: reg ystar_all probable_risk_2030_share  [aw=w_all], vce(robust)
// eststo: reg d_ystar flood_risk_share  [aw=w], vce(robust)
// eststo: reg d_ystar probable_risk_2030_share  [aw=w], vce(robust)
//
// esttab using "$tab/climate_risk.tex", ///
// 	mgroups("$ y^*$" "$\Delta y^*$", pattern(1 0 1) ///
// 				prefix(\multicolumn{@span}{c}{) suffix(}) ///
// 				span erepeat(\cmidrule(lr){@span})) ///
// 	nomtitle ///
// 	keep(flood_risk_share probable_risk_2030_share) ///
// 	varlabel(	flood_risk_share "Flood Risk" ///
// 				probable_risk_2030_share "Subsidence Risk") ///
// 	stats(N r2, label("N" "R2") fmt(0 2)) ///
// 	se(2) b(2) replace
