* Expand Hilber & Vermeulen data set on house prices and income to the present

********** Get inflation deflator **********
// Use OECD CPI in early years where missing retail price index
import delimited "$raw/fred/CPGRLE01GBM659N.csv", clear
gen year=year(date(date, "YMD"))
collapse pct_change=cpg, by(year)
tempfile cpi 
save `cpi'

// USE retail price index when it is available, per Hilber & Vermeulen
import delimited "$raw/ons/inflation/series-100224.csv", clear 
keep if _n>=9 & _n<=56
rename (v1 v2) (year pct_change) 
destring, replace
merge 1:1 year using `cpi', nogen

gen rpi = 1 if year==2008
replace rpi = rpi[_n-1] * (1 + pct_change/100) if year>2008

gsort -year 
replace rpi = rpi[_n-1] / (1 + pct_change/100) if year<2008
keep year rpi
save "$working/rpi.dta", replace

********** House Prices **********
use "$working/merged_hmlr.dta", clear
merge m:1 postcode using "$raw/geography/lpa_codes.dta", nogen keep(match)
keep if strpos(lpa_code, "E")==1

// Weight to remove differences in type composition over time
gegen N = count(log_price), by(lpa_code)
gegen N_i = count(log_price), by(lpa_code type)
gen sh_i = N_i/N

gegen N_it = count(log_price), by(lpa_code type year)
gegen N_t = count(log_price), by(lpa_code year)
gen sh_it = N_it/N_t 

gen w_it = sh_i/sh_it
gcollapse price [aw=w_it], by(lpa_code year)

gsort lpa_code year
by lpa_code: gen hpi = price/price[1]
tempfile hmlr_price_index
save `hmlr_price_index'

// Merge with Hilber & Vermeulen data
use "$clean/hilber_lad21.dta", clear
merge 1:1 lpa_code year using `hmlr_price_index'

// Make Hilber & Vermeulen data nominal
merge m:1 year using  "$working/rpi.dta", keep(match) nogen
gen rpi1974 = rpi if year==1974 
ereplace rpi1974=mean(rpi1974), by(lpa_code)

gen hilber_index = rindex2*(rpi/rpi1974) // Get nominal hilber index
gen hilber_index1995 = hilber_index if year==1995 
ereplace hilber_index1995=mean(hilber_index1995), by(lpa_code)

// Create expanded index 
replace hpi = hilber_index if year<1995 
replace hpi = hilber_index1995 * hpi if year>=1995

keep lpa_code year hpi
gen log_hpi = log(hpi/100)
save "$working/lpa_hpi.dta", replace

********** Income **********
use "$working/ashe_earnings.dta", clear
drop Description
drop if year==2022 // Data for 2022 is preliminary and not available for all LAs
reshape wide earn num_jobs, i(lpa_code) j(year)
tempfile ashe_wide 
save `ashe_wide'

use "$clean/hilber_lad21.dta", clear
keep male_earn_real rindex2 lpa_code year
reshape wide male_earn_real rindex2, i(lpa_code) j(year)

merge 1:1 lpa_code using  `ashe_wide', keep(match) nogen
reshape long male_earn_real rindex2 earn num_jobs, i(lpa_code) j(year)

********** Merge all **********
merge 1:1 lpa_code year using "$working/lpa_hpi.dta", keep(match) nogen
merge m:1 year using "$working/rpi.dta", keep(master match) nogen

// Extend earnings series with Hilber & Vermeulen data when missing
gen male_earn_nom = male_earn_real*rpi
replace earn = male_earn_nom if missing(earn)
gen log_earn = log(earn)

save "$clean/expanded_hilber_data.dta", replace

