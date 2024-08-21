******************************************************************
* Code for: "Measuring the Natural Rate With Natural Experiments"
* Backer-Peral, Hazell, Mian (2023)
* 
* This program calculates the extension hazard rate.
******************************************************************

use "$clean/leasehold_flats.dta", clear  

* Infer how many extensions we are missing:
distinct property_id if closed_lease & number_years==99 & round(extension_amount)==90 // Total number of recorded 189 extensions with pre-transaction data
local tot = r(ndistinct)
distinct property_id if number_years == 189 & L_date_trans==. // 189 extensions that were not recorded
local missing_pre = r(ndistinct)
global hazard_correction = 1 + `missing_pre'/`tot'

global startyear = 2003

* Round values
gen extension_year = year(date_extended) if has_been_extended | has_extension

replace extension_amount = int(extension_amount)
replace extension_amount = 90 if has_been_extended & !extension
replace duration = int(duration)
drop duration????

* Keep only one entry per property. For extensions, make sure it's the extended entry 
drop if has_extension & !extension
gegen tag = tag(property_id)
keep if tag==1

* Get starting duration 
gen duration$startyear = duration + (year - $startyear)
replace duration$startyear = duration$startyear - extension_amount if date_extended<=date_trans & year(date_extended)>$startyear & extension

gen extended$startyear = extension_year==$startyear
drop if duration$startyear<0	

forvalues year = $startyear/2023 {
	
	di "`year'"
	
	local prev_year = `year'-1
	
	if `year' != $startyear {
		* In general, the duration decreases by 1 every year
		qui: gen duration`year' = duration`prev_year' - 1
		
		* If the lease was extended or registered for the first time in this year, update accordingly
		qui: replace duration`year' = extension_amount + duration`year' if extension_year == `year'
		
		* Also, identify this as a least extension if it is one
		qui: gen extended`year' = 1 if extension_year == `year'
	}
}

// Drop anything that goes negative at some point and then was extended since we don't know what these are
egen mindur = rowmin(duration????)
drop if mindur<0 & (has_been_extended | extension)

keep property_id extended???? duration???? number_years date_from date_extended extension_amount extension has_been_extended
cap drop duration*yr

gegen property_id_n = group(property_id)

compress
greshape long duration extended, i(property_id property_id_n number_years date_from date_extended extension_amount) j(year)

replace duration=. if year < year(date_from) & !(extension | has_been_extended)
drop if duration<0 | missing(duration)

replace extended = 0 if missing(extended)

gsort property_id year
by property_id: gen L_duration = duration[_n-1]
gen L_duration5yr = round(L_duration, 5)
drop if L_duration == 0

by property_id: gen previously_extended = sum(extended)
egen has_extension = total(extended), by(property_id)
gen will_be_extended = has_extension - previously_extended

gen whb_duration = duration
replace whb_duration = duration - extension_amount if previously_extended

gen L_duration_bin = int(L_duration) if L_duration>=70
replace L_duration_bin = L_duration5yr if L_duration<70

save "$clean/leasehold_panel.dta", replace

// Get hazard rate for post-period by duration
keep if year>=2010 & year<2020
collapse (mean) extended, by(L_duration)

keep if L_duration>=80
gen hazard_rate = extended * $hazard_correction

gsort L_duration
gen inv_hazard = 1-hazard
gen prod_inv_hazard = inv_hazard if _n==1
replace prod_inv_hazard = inv_hazard * prod_inv_hazard[_n-1] if _n>1
gen cum_prob = 100 * (1-prod_inv_hazard)

keep L_duration cum_prob 
rename L_duration whb_duration 
save "$clean/hazard_rate.dta", replace
