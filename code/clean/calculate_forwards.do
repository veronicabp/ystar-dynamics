**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

* Import data on nominal interest rates (Download here: https://www.bankofengland.co.uk/statistics/yield-curves)
import excel "$raw/boe/glcnominalmonthedata/GLC Nominal month end data_1970 to 2015.xlsx", clear sheet("4. spot curve")
tempfile temp
save `temp'
import excel "$raw/boe/glcnominalmonthedata/GLC Nominal month end data_2016 to present.xlsx", clear sheet("4. spot curve")
append using `temp'

gen date = date(A, "DMY")

foreach var of varlist A-CC {
	cap local num = `var'[4]
	cap local s = subinstr("`num'", ".00","",.)
	di "`s'"
	cap rename `var' y`s'
}

keep date y? y??

keep date y25 y30 y10 y5 y1
gen year = year(date)
gen month = month(date)
rename y?? uk??y
rename y? uk?y

drop if year==.

tempfile nominal
save `nominal'

* Import data on real interest rates 
import excel "$raw/boe/glcrealmonthedata/GLC Real month end data_1979 to 2015.xlsx", clear sheet("4. spot curve")
tempfile temp
save `temp'
import excel "$raw/boe/glcrealmonthedata/GLC Real month end data_2016 to present.xlsx", clear sheet("4. spot curve")
append using `temp'

gen date = date(A, "DMY")

foreach var of varlist A-IR {
	cap local num = `var'[4]
	cap local s = subinstr("`num'", ".00","",.)
	di "`s'"
	cap rename `var' y`s'
}

keep date y? y??

keep date y5 y25 y30 y10
gen year = year(date)
gen month = month(date)
rename y?? uk??y_real
rename y? uk?y_real

drop if year==.

* Merge both data sets
merge 1:1 date using `nominal', nogen

sort date
format date %tdDD-NN-CCYY

* Calculate forward rates
gen uk10y20 = 100 * ((((1+uk30y/100)^30)/((1+uk10y/100)^10))^(1/20) - 1)
gen uk10y15 = 100 * ((((1+uk25y/100)^25)/((1+uk10y/100)^10))^(1/15) - 1)
gen uk10y15_real = 100 * ((((1+uk25y_real/100)^30)/((1+uk10y_real/100)^10))^(1/20) - 1)
gen uk10y20_real = 100 * ((((1+uk30y_real/100)^25)/((1+uk10y_real/100)^10))^(1/15) - 1)

save "$clean/uk_interest_rates.dta", replace
