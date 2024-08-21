**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

********************************************
* Rent panel
********************************************

use "$clean/leasehold_flats.dta", clear 
keep property_id date_trans L_date_trans
tempfile transactions 
save `transactions'

* For all Rental data, de-mean rental growth by year of listing

// Rightmove
use "$working/rightmove_rents_flats.dta", clear
joinby property_id_rm using  "$working/rightmove_merge_keys"

rename listingprice rent  
gen log_rent = log(rent)

// Remove outliers
// winsor rent, gen(win) p(0.01)
// keep if rent==win

collapse log_rent, by(property_id date_rm)
format date_rm %td

tempfile rightmove 
save `rightmove'

// Zoopla
use "$working/zoopla_rent.dta", clear
joinby property_id_zoop using  "$working/zoopla_merge_keys"
rename price_zoopla rent 
drop if missing(date_zoopla)

replace rent = rent*52 // annualize
gen log_rent = log(rent)

// Remove outliers
// winsor rent, gen(win) p(0.01)
// keep if rent==win

collapse log_rent, by(property_id date_zoop)
rename date_zoop date_rm
format date_rm %td

tempfile zoopla 
save `zoopla'

use `zoopla', clear
append using `rightmove'

drop if missing(log_rent)
collapse log_rent, by(property_id date_rm) // If there's a Rightmove and Zoopla listing on the same day, take mean

gsort property_id date_rm
gen year_rm = year(date_rm)

// Restrict to properties with more than one listing 
bys property_id: gen N=_N
drop if N<=1
drop N

// Take differences
gsort property_id date_rm 
by property_id: gen L_date_rm = date_rm[_n-1]
by property_id: gen L_year_rm = year_rm[_n-1]
by property_id: gen L_log_rent = log_rent[_n-1]
gen d_log_rent = log_rent - L_log_rent 
gen d_log_rent_ann = d_log_rent/((date_rm-L_date_rm)/365)

// Drop pairs that are within 6 months of each other
drop if datediff(L_date_rm, date_rm, "month")<6

// Merge in whether there is a transaction between 
joinby property_id using `transactions'

gen has_trans = L_date_rm<=date_trans & date_rm>=date_trans
ereplace has_trans=max(has_trans), by(property_id date_rm)
drop date_trans L_date_trans 
gduplicates drop property_id date_rm, force

reghdfe d_log_rent i.has_trans, absorb(year_rm##L_year_rm) residuals(d_log_rent_res)
gegen mn=mean(d_log_rent), by(year_rm L_year_rm)
replace d_log_rent_res = d_log_rent_res + mn 
drop mn

save "$working/log_rent_panel.dta", replace

***** Export data for repeat sales index
* Get necessary variables
use "$clean/leasehold_flats.dta", clear 
drop if has_extension & !extension
keep property_id duration2023 latitude longitude area extension date_extended
gduplicates drop property_id, force
tempfile vars 
save `vars'

use "$working/log_rent_panel.dta", clear
format *date* %tdDD-NN-CCYY

* Merge in necessary variables
merge m:1 property_id using `vars', keep(match) nogen 
export delimited "$working/for_rent_rsi.csv", replace

*********************************************
* Get experiment IDs

// Extensions 
use "$clean/experiments.dta", clear
cap drop date

gen experiment_pid = property_id 
gen experiment_date = date_trans
gen date = date_trans

format *date %td

keep experiment_pid experiment_date property_id date
gen type = "extension"
save "$working/extension_pids.dta", replace

// Controls
import delimited "$working/rsi/rsi_residuals.csv", clear

foreach var of varlist date_trans_control date_trans_treated {
	cap drop temp 
	gen temp = date(`var', "MDY")
	drop `var'
	rename temp `var'
}
format date* %td

gen experiment_pid = pid_treated 
gen experiment_date = date_trans_treated

gen property_id = pid_control 
gen date = date_trans_control

keep experiment_pid experiment_date property_id date
gen type = "control"
save "$working/control_pids.dta", replace

// Append both
use "$working/extension_pids.dta", clear
append using "$working/control_pids.dta"
gsort experiment_pid experiment_date type
save "$working/experiment_pids.dta", replace

*********************************************
* Merge rent data with treatment/control ids

* Get extensions which have data 
use "$working/extension_pids.dta", clear
joinby property_id using "$working/log_rent_panel.dta"
keep experiment_pid experiment_date 
gduplicates drop 
tempfile matched_pids 
save `matched_pids'

* Get data for rent repeat sales index (using same control properties)
use "$working/experiment_pids.dta", clear
merge m:1 experiment_pid experiment_date  using `matched_pids', keep(match) nogen

* Keep only one instance of each property per experiment 
drop date
gegen experiment=group(experiment_pid experiment_date)
gduplicates drop experiment property_id type, force

* Merge with rent data 
joinby property_id using "$working/log_rent_panel.dta"

* Merge in data on the experiment 
rename property_id pid 
rename experiment_pid property_id 
rename experiment_date date_trans 
merge m:1 property_id date_trans using "$clean/experiments.dta", keep(match) keepusing(year L_year L_date_trans date_extended) nogen

save "$working/experiment_rent_panel.dta", replace

********************************************
* r_K time series
********************************************
use "$clean/experiments.dta", clear
foreach freq in "yearly" "quarterly" "monthly" {
	foreach tag in "" {
		gen ystar_`freq'`tag' = .
		gen ub_`freq'`tag' = .
		gen lb_`freq'`tag' = .
	}
}

gen xaxis12 = $year0*12 + _n - 1 if $year0*12 + _n <= $year1 * 12 + $month1
gen xaxis = (xaxis12)/12
gen q = mod(int((_n-1)/3), 4) + 1

forv year=$year0 / $year1 {
	di `year'

	qui: nl $nlfunc if year==`year', initial(ystar 3) variables(did_rsi T k) vce(robust)
	qui: replace ystar_yearly = _b[/ystar] if int(xaxis)==`year'
	qui: replace ub_yearly = _b[/ystar] + 1.96*_se[/ystar] if int(xaxis)==`year'
	qui: replace lb_yearly = _b[/ystar] - 1.96*_se[/ystar] if int(xaxis)==`year' 
	
	forv quarter = 1/4 {
		qui: sum did_rsi if year==`year' & quarter==`quarter'
		if r(N)<=50 continue
		
		qui: nl $nlfunc if year==`year' & quarter==`quarter', initial(ystar 3) variables(did_rsi T k) vce(robust)
		qui: replace ystar_quarterly = _b[/ystar] if int(xaxis)==`year' & q==`quarter' & abs(_b[/ystar]) < 10
		qui: replace ub_quarterly = _b[/ystar] + 1.96*_se[/ystar] if int(xaxis)==`year' & q==`quarter' & abs(_b[/ystar]) < 10
		qui: replace lb_quarterly = _b[/ystar] - 1.96*_se[/ystar] if int(xaxis)==`year' & q==`quarter' & abs(_b[/ystar]) < 10
		
		forv month = 1/3 {
			local month = (`quarter'-1)*3 + `month'
			di " `month'"
			
			local date = `year'*12 + `month' - 1
			
			qui: sum did_rsi if year==`year' & month==`month'
			if r(N)<=50 {
// 				di "Insufficient obs."
				continue
			}
			
			qui: nl $nlfunc if year==`year' & month==`month', initial(ystar 3) variables(did_rsi T k)
			qui: replace ystar_monthly = _b[/ystar] if xaxis12==`date' & abs(_b[/ystar]) < 10
			qui: replace ub_monthly = _b[/ystar] + 1.96*_se[/ystar] if xaxis12==`date' & abs(_b[/ystar]) < 10
			qui: replace lb_monthly = _b[/ystar] - 1.96*_se[/ystar] if xaxis12==`date' & abs(_b[/ystar]) < 10		
		}
	}
}

keep xaxis12 xaxis q ystar* ub* lb*
drop if xaxis==.

gen year = int(xaxis)
gen month = round((xaxis-year)*12) + 1
rename q quarter

foreach freq in "year" "quarter" "month" {
	preserve
		keep ystar_`freq'ly ?b_`freq'ly year quarter month xaxis12
		rename *_`freq'ly * 
		duplicates drop ystar year, force
		
		gen var = ((ub-ystar)/1.96)^2
		
		save "$clean/ystar_`freq'ly.dta", replace
	restore
}

*************************************
* Data for event study
*************************************

* Get index
import delimited "$working/rsi/rsi_full.csv", clear
gen date_trans_ext=date(date_trans, "MDY")
drop date_trans
gduplicates drop property_id date, force
tempfile rsi_full_nd
save `rsi_full_nd'

* Get property ids for properties that extend
use "$clean/experiments.dta", clear 
drop if missing(did_rsi)
keep property_id year whb_duration
rename year experiment_year
rename whb_duration experiment_duration
duplicates drop property_id, force
save "$working/pids.dta", replace

use "$clean/leasehold_flats.dta", clear
merge m:1 property_id using "$working/pids.dta", keep(match) nogen
drop if year==L_year & quarter==L_quarter

// Drop properties that extend multiple times
drop if multiple_extensions

cap drop date 
gen date = year*4 + quarter

merge 1:1 property_id date using `rsi_full_nd', keep(match) nogen

replace extension_amount = . if !extension 
ereplace extension_amount = mean(extension_amount), by(property_id)

save "$working/for_event_study.dta", replace

*************************************
* Calculate housing risk premium
*************************************
import delimited "$raw/fred/UKNGDP.csv", clear 
rename (date ukngdp) (d gdp) 
gen date = date(d, "YMD")
gen year = year(date)
gen quarter = quarter(date)

tempfile gdp
save `gdp'

* Use 2022 levels to back out all previous levels of price/rent
use "$clean/leasehold_flats.dta", clear
sum price if year==2022 & quarter==4
scalar price2022Q4 = r(mean)

sum rent if year(date_rent)==2022 & quarter(date_rent)==4
scalar rent2022Q4 = r(mean)

import delimited "$raw/oecd/house_prices.csv", clear 
keep if location=="GBR" & frequency=="Q"

gen date = dofq(quarterly(time, "YQ"))
gen year = year(date)
gen quarter = quarter(date)
keep if date <= 22919

keep value year quarter date subject 
reshape wide value, i(year quarter date) j(subject) string 
rename (valueNOMINAL valueRENT) (price_index rent_index)
keep year quarter date price_index rent_index

gsort -date
foreach var in price rent {
	gen `var' = `var'2022Q4 if _n==1
	replace `var' = `var'[_n-1] * `var'_index[_n]/`var'_index[_n-1] if _n>1
}

gen rent_price = rent/price

merge 1:1 year quarter using `gdp', keep(match) nogen

tempfile temp
save `temp'

use "$clean/uk_interest_rates.dta", clear 
gen quarter = quarter(date)
collapse uk10y15 uk10y, by(year quarter)

merge 1:1 year quarter using `temp', keep(match) nogen
format date %tdNN-CCYY

cap drop d
gen d = qofd(date)
tsset d

* Define variables
sort year
gen g = (rent - rent[_n-1])/rent[_n-1]
gen r = uk10y15/100
gen d_gdp = (gdp-gdp[_n-1])/gdp[_n-1]

var g rent_price d_gdp

// Compute 30y ahead forecast for every year
local T = 30*4
forv y=160/251 {
	fcast compute f`y'_, step(`T') dynamic(`y')
}

// For each, compute mean balanced growth 
gen g_balanced = .
gen rtp_balanced = .
forv y=160/251 {
	qui: sum f`y'_g
	qui: replace g_balanced = r(mean) if d==`y'
	
	qui: sum f`y'_rent_price
	qui: replace rtp_balanced = r(mean) if d==`y'
}

replace rtp_balanced = 100 * rtp_balanced
replace g_balanced = 100 * g_balanced
replace r = 100 * r
gen risk_premium = rtp_balanced + g_balanced - r

gen date_q = d 
format date_q %td 
line risk_premium g_balanced r rtp_balanced  date_q if !missing(risk_premium)

keep risk_premium g_balanced year quarter 
drop if missing(risk_premium)
save "$clean/risk_premium.dta", replace

********** Calculate forward rate for multiple countries *********

// International GDP
import delimited "$raw/oecd/gdp.csv", clear 
keep if measure=="MLN_USD"
keep if time>=2010 & time<=2020 // Keep years for which we have data for all countries
collapse gdp=value, by(location)
rename location iso
tempfile gdp 
save `gdp'

// 10 Year
import excel "$raw/global_financial_data/rate10yr.xlsx", sheet(Data Information) clear firstrow
keep Ticker Country
tempfile info 
save `info'

import excel "$raw/global_financial_data/rate10yr.xlsx", sheet(Price Data) clear firstrow
merge m:1 Ticker using `info', nogen
rename Close rate10y
tempfile r10y 
save `r10y'

import excel "$raw/global_financial_data/rate30yr.xlsx", sheet(Data Information) clear firstrow
keep Ticker Country
tempfile info 
save `info'

import excel "$raw/global_financial_data/rate30yr.xlsx", sheet(Price Data) clear firstrow
merge m:1 Ticker using `info', nogen
rename Close rate30y

merge 1:1 Date Country using `r10y'

gen date=date(Date, "MDY")
rename Country country 

keep country date rate* 
gsort country date 
format date %td
gen year=year(date)
gen quarter=quarter(date)

replace rate30y = rate30y/100
replace rate10y = rate10y/100

gen rate10y20 = 100 * ((((1+rate30y/100)^30)/((1+rate10y/100)^10))^(1/20) - 1)
drop if missing(rate10y20)

merge m:1 country using "$raw/global_financial_data/iso_codes.dta", nogen
merge m:1 iso using `gdp', nogen keep(match)

// Keep countries that have date in every year 
levelsof iso, local(isos)
foreach iso of local isos {
	forv year=2003/2023 {
		qui: sum rate10y20 if year==`year' & iso=="`iso'"
		if r(N)==0 {
			di "Dropping `iso'"
			qui: drop if iso=="`iso'"
		}
	}
}

tab country

collapse rate10y20 (first) date [aw=gdp], by(year)
save "$working/global_forward.dta", replace 

****** Get global measures of valuations ******

// Rent-to-price ratio
import delimited "$raw/oecd/house_prices.csv", clear
keep if frequency=="Q" 
keep if subject=="PRICERENT"

gen date = quarterly(time, "YQ")
gen rent_price = 100 * (1/(value/100))
keep date rent_price location
rename location iso

gen year = year(dofq(date))
collapse rent_price, by(year iso)

// Keep countries for which there is data in every year of sample
levelsof iso, local(isos)
foreach iso of local isos {
	forv year=2003/2023 {
		qui: sum rent_price if year==`year' & iso=="`iso'"
		if r(N)==0 {
			di "Dropping `iso'"
			qui: drop if iso=="`iso'"
		}
	}
}


merge m:1 iso using `gdp', nogen keep(match)
drop if iso=="OECD"

preserve 
	keep if iso=="GBR"
	keep rent_price year 
	rename rent_price rent_price_uk
	save "$working/uk_rtp.dta", replace
restore

collapse rent_price [aw=gdp], by(year)

tempfile rent_price 
save `rent_price'

gen date=year

keep rent_price year 
rename rent_price rent_price_global
save "$working/global_rtp.dta", replace


******************** Save rightmove descriptions that are matched to HMLR flats 
use "$working/rightmove_descriptions.dta", clear 
merge m:1 property_id_rm using "$working/rightmove_merge_keys", keep(match) nogen
replace summary = strtrim(summary)
replace summary = upper(summary)
drop if missing(summary)
keep property_id* date_rm summary

gen renovated = strpos(summary, "RENOVATED") | ///
				strpos(summary, "REFURBISHED") | ///
				strpos(summary, "UPDATED") | ///
				strpos(summary, "IMPROVED") | ///
				strpos(summary, "NEWLY BUILT") | ///
				strpos(summary, "NEW BEDROOM") | ///
				strpos(summary, "NEW BATHROOM") | ///
				strpos(summary, "NEW KITCHEN")
				
gcollapse (max) renovated, by(property_id date_rm)				
save "$working/rightmove_descriptions_matched.dta", replace

// Link renovations to transactions
use "$clean/leasehold_flats.dta", clear
joinby property_id using "$working/rightmove_descriptions_matched.dta"
gen d=date_trans - date_rm
keep if abs(d)<365*2 // Keep listings within 2 years of transaction time
gegen mind = min(d), by(property_id date_trans)
keep if d==mind  // Keep listing closest to transaction time
save "$working/renovations.dta", replace 
