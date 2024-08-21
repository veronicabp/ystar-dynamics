******************************************************************
* Code for: "Measuring the Natural Rate With Natural Experiments"
* Backer-Peral, Hazell, Mian (2023)
* 
* This program merges Land Registry data with all other data sets
* and produces a final data set of all leasehold flats.
******************************************************************

**************************************
* Merge hedonics
**************************************
use "$working/merged_hmlr.dta", clear
keep if flat

joinby property_id using  "$working/rightmove_merge_keys", unmatched(master)
drop _merge

************** Rightmove rents **************
joinby property_id_rm using "$working/rightmove_rents_flats.dta", unmatched(master)
drop _merge

drop listprice0 listprice1 datelist0 datelist1
rename listingid rentid
rename listingprice rent 

// Keep the closest listing for each hedonic variable
foreach var of varlist $hedonics_rm_full rent {
	di "`var'"
	
	qui: gen diff = abs(date_trans - date_rm) if !missing(`var')
	qui: gegen mindiff = min(diff), by(property_id date_trans)
	
	qui: gen date_`var'_closest = date_rm if diff==mindiff & mindiff!=.
	qui: gen `var'_closest = `var' if diff==mindiff

	qui: gegen date_`var'_closest = firstnm(date_`var'_closest), by(property_id date_trans) replace
	qui: gegen `var'_closest = firstnm(`var'_closest), by(property_id date_trans) replace
	
	drop diff mindiff
}

drop $hedonics_rm_full rent date_rm
rename *_closest *
gduplicates drop property_id date_trans property_id_rm, force

// Rename info so that it doesn't get overwritten by sales data
foreach var of varlist $hedonics_rm_full {
	rename `var' `var'_rent
	rename date_`var' date_`var'_rent
}

************** Rightmove hedonics **************
joinby property_id_rm using "$working/rightmove_sales_flats.dta", unmatched(master)
drop _merge

// Keep the closest listing for each hedonic variable
foreach var of varlist $hedonics_rm_full {
	di "`var'"
	
	qui: gen diff = abs(date_trans - date_rm) if !missing(`var')
	qui: gegen mindiff = min(diff), by(property_id date_trans)
	
	qui: gen date_`var'_closest = date_rm if diff==mindiff & mindiff!=.
	qui: gen `var'_closest = `var' if diff==mindiff

	qui: gegen date_`var'_closest = firstnm(date_`var'_closest), by(property_id date_trans) replace
	qui: gegen `var'_closest = firstnm(`var'_closest), by(property_id date_trans) replace
	
	drop diff mindiff
}

drop $hedonics_rm_full date_rm
rename *_closest *
gduplicates drop property_id date_trans, force

// Use rental data hedonics when missing listing hedonics
foreach var of varlist $hedonics_rm {
	replace date_`var' = date_`var'_rent if missing(`var')
	replace `var' = `var'_rent if missing(`var')
}

// Identify how long a house was on the market for and how it's price changed
gen time_on_market = datelist1-datelist0
gen price_change_pct = 100 * (listprice1 - listprice0)/listprice0

************** Zoopla rent **************
joinby property_id using "$working/zoopla_merge_keys.dta", unmatched(master)
drop _merge

joinby property_id_zoop using "$working/zoopla_rent.dta", unmatched(master)
drop _merge
rename price_zoop rent_zoop

// Keep the closest listing for each hedonic variable
foreach var of varlist $hedonics_zoop rent_zoop {
	di "`var'"
	
	qui: gen diff = abs(date_trans - date_zoop) if !missing(`var')
	qui: gegen mindiff = min(diff), by(property_id date_trans)
	
	qui: gen date_`var'_closest = date_zoop if diff==mindiff & mindiff!=.
	qui: gen `var'_closest = `var' if diff==mindiff

	qui: gegen date_`var'_closest = firstnm(date_`var'_closest), by(property_id date_trans) replace
	qui: gegen `var'_closest = firstnm(`var'_closest), by(property_id date_trans) replace
	
	drop diff mindiff
}

drop $hedonics_zoop rent_zoop date_zoop
rename *_closest * 
gduplicates drop property_id date_trans property_id_zoop, force

// Annualize rent
replace rent_zoop = rent_zoop*52

foreach var of varlist $hedonics_zoop {
	rename `var' `var'_rent
	rename date_`var' date_`var'_rent
}

************** Merge in Zoopla hedonics **************
joinby property_id_zoop using "$working/zoopla_sales.dta", unmatched(master)
drop _merge

// Keep the closest listing for each hedonic variable
foreach var of varlist $hedonics_zoop {
	di "`var'"
	
	qui: gen diff = abs(date_trans - date_zoop) if !missing(`var')
	qui: gegen mindiff = min(diff), by(property_id date_trans)
	
	qui: gen date_`var'_closest = date_zoop if diff==mindiff & mindiff!=.
	qui: gen `var'_closest = `var' if diff==mindiff

	qui: gegen date_`var'_closest = firstnm(date_`var'_closest), by(property_id date_trans) replace
	qui: gegen `var'_closest = firstnm(`var'_closest), by(property_id date_trans) replace
	
	drop diff mindiff
}

drop $hedonics_zoop date_zoop
rename *_closest *
gduplicates drop property_id date_trans, force

// Use rental data hedonics when missing listing hedonics
foreach var of varlist $hedonics_zoop {
	replace date_`var' = date_`var'_rent if missing(`var')
	replace `var' = `var'_rent if missing(`var')
}

// Sanity check: zoopla and rightmove bedroom count should line up (removing extreme outliers)
corr bedrooms bedrooms_zoop  if bedrooms<=5 & bedrooms_zoop <=5

// Replace hedonics with zoopla hedonics if missing rightmove hedonics
foreach var of varlist bedrooms bathrooms {
	replace date_`var' = date_`var'_zoop if (`var'==0 | `var'==.)
	replace `var' = `var'_zoop if (`var'==0 | `var'==.)
}

replace date_rent = date_rent_zoop if missing(rent) & !missing(rent_zoop)
replace rent = rent_zoop if missing(rent)

gen age = year(date_trans) - yearbuilt
replace yearbuilt = . if age < 0
replace age = . if age < 0

****************************************************************************
* Remove erroneous rent values 
****************************************************************************
replace rent = . if rent<=100 | (rent > 0.7*price & !missing(rent))

gen bedrooms_z = bedrooms if bedrooms <=10
gen bathrooms_z = bathrooms if bathrooms <=5

gen log_rent = log(rent)
gen log_rent100 = 100 * log_rent

fasterxtile floorarea_50=floorarea, nq(50)
fasterxtile yearbuilt_50=yearbuilt, nq(50)
fasterxtile age_50=age, nq(50)

// Get dummies of hedonics
// Make dummies of hedonics
egen condition_n  = group(condition), mi
egen heating_n    = group(heatingtype), mi
egen parking_n    = group(parking), mi

// GMS-style hedonics
gen bedrooms_n = bedrooms
gen bathrooms_n = bathrooms
gen livingrooms_n = livingrooms

replace bedrooms_n     = 99 if bedrooms     == . | bedrooms > 8
replace bathrooms_n    = 99 if bathrooms    == . | bathrooms > 8
replace livingrooms_n  = 99 if livingrooms  == . | livingrooms > 8

fasterxtile floorarea_n=floorarea, nq(50)
replace floorarea_n = 99 if missing(floorarea_n)

fasterxtile age_n=age, nq(50)
replace age_n = 99 if missing(age_n)

*********************************
* Residualize price on hedonics 
*********************************
foreach var of varlist bedrooms bathrooms floorarea_50 yearbuilt_50 {
	qui: reghdfe log_price, absorb(`var') residuals(pres_`var')
	qui: sum log_price if e(sample)
	qui: replace pres_`var' = pres_`var' + r(mean) 	
}

// Main hedonics
reghdfe log_price, absorb(bedrooms bathrooms floorarea_50 yearbuilt_50) residuals(pres_main)
sum log_price if e(sample)
replace pres_main = pres_main + r(mean) 

// All 
reghdfe log_price, absorb(bedrooms bathrooms floorarea_50 yearbuilt_50 condition heatingtype parking) residuals(pres_all)
sum log_price if e(sample)
replace pres_all = pres_all + r(mean) 

// All (GMS style)
reghdfe log_price, absorb(bedrooms_n bathrooms_n floorarea_n age_n condition_n heating_n parking_n) residuals(pres_all_gms) // all
sum log_price if e(sample)
replace pres_all_gms = pres_all_gms + r(mean) 

// Linear controls
reghdfe log_price bedrooms floorarea, residuals(log_price_res) noabsorb
sum log_price if e(sample)
replace log_price_res = log_price_res + r(mean) 

gen pres_linear = log_price_res		

// Quadratic controls
gen bedrooms2 = bedrooms * bedrooms 
gen floorarea2 = floorarea * floorarea

reghdfe log_price bedrooms floorarea bedrooms2 floorarea2, residuals(pres_quad) noabsorb
sum log_price if e(sample)
replace pres_quad = pres_quad + r(mean) 

****** Interact with year FE
// Main hedonics
reghdfe log_price, absorb(bedrooms##year bathrooms##year floorarea_50##year yearbuilt_50##year) residuals(tpres_main)
cap drop log_price_samp mn
qui: gen log_price_samp = log_price if e(sample)
qui: gegen mn = mean(log_price_samp), by(year)
qui: replace tpres_main = tpres_main + mn

// All 
reghdfe log_price, absorb(bedrooms##year bathrooms##year floorarea_50##year yearbuilt_50##year condition##year heatingtype##year parking##year) residuals(tpres_all)
cap drop log_price_samp mn
qui: gen log_price_samp = log_price if e(sample)
qui: gegen mn = mean(log_price_samp), by(year)
qui: replace tpres_all = tpres_all + mn

// All (GMS style)
reghdfe log_price, absorb(bedrooms_n##year bathrooms_n##year floorarea_n##year age_n##year condition_n##year heating_n##year parking_n##year) residuals(tpres_all_gms) // all
cap drop log_price_samp mn
qui: gen log_price_samp = log_price if e(sample)
qui: gegen mn = mean(log_price_samp), by(year)
qui: replace tpres_all_gms = tpres_all_gms + mn

// Linear controls
reghdfe log_price c.bedrooms##i.year c.floorarea##i.year, residuals(tpres_linear) noabsorb
cap drop log_price_samp mn
qui: gen log_price_samp = log_price if e(sample)
qui: gegen mn = mean(log_price_samp), by(year)
qui: replace tpres_linear = tpres_linear + mn

// Quadratic controls
reghdfe log_price (c.bedrooms c.floorarea c.bedrooms2 c.floorarea2)##i.year, residuals(tpres_quad) noabsorb
cap drop log_price_samp mn
qui: gen log_price_samp = log_price if e(sample)
qui: gegen mn = mean(log_price_samp), by(year)
qui: replace tpres_quad = tpres_quad + mn

drop bedrooms2 floorarea2

// Now get all combinations of the main hedonic characteristics
cap drop mn
local count=1
local varlist "bedrooms bathrooms floorarea_50 yearbuilt_50 condition heatingtype parking"
forv a = 1/7 {
	local ap1 = `a'+1
	local var : word `a' of `varlist'
	local fe1 "`var'"
	di "`fe1'"
	qui: reghdfe log_price, absorb(`fe1') residuals(pres`count') 
	qui: sum log_price if e(sample)
	qui: replace pres`count' = pres`count' + r(mean) 	
	
	local count = `count'+1
	
	//
	forv b = `ap1'/7 {
		local bp1 = `b'+1
		local var : word `b' of `varlist'
		local fe2 "`fe1' `var'"
		di "`fe2'"
		qui: reghdfe log_price, absorb(`fe2') residuals(pres`count') 
		qui: sum log_price if e(sample)
		qui: replace pres`count' = pres`count' + r(mean) 	
		
		local count = `count'+1
		
		//
		forv c = `bp1'/7 {
			local cp1 = `c'+1
			local var : word `c' of `varlist'
			local fe3 "`fe2' `var'"
			di "`fe3'"
			qui: reghdfe log_price, absorb(`fe3') residuals(pres`count') 
			qui: sum log_price if e(sample)
			qui: replace pres`count' = pres`count' + r(mean) 	
			
			local count = `count'+1
			
			//
			forv d = `cp1'/7 {
				local dp1 = `d'+1
				local var : word `d' of `varlist'
				local fe4 "`fe3' `var'"
				di "`fe4'"
				qui: reghdfe log_price, absorb(`fe4') residuals(pres`count') 
				qui: sum log_price if e(sample)
				qui: replace pres`count' = pres`count' + r(mean) 	
				
				local count = `count'+1
				
				//
				forv e = `dp1'/7 {
					local ep1 = `e'+1
					local var : word `e' of `varlist'
					local fe5 "`fe4' `var'"
					di "`fe5'"
					qui: reghdfe log_price, absorb(`fe5') residuals(pres`count') 
					qui: sum log_price if e(sample)
					qui: replace pres`count' = pres`count' + r(mean) 	
					
					local count = `count'+1
					
					//
					forv f = `ep1'/7 {
						local fp1 = `f'+1 
						local var : word `f' of `varlist'
						local fe6 "`fe5' `var'"
						di "`fe6'"
						qui: reghdfe log_price, absorb(`fe6') residuals(pres`count') 
						qui: sum log_price if e(sample)
						qui: replace pres`count' = pres`count' + r(mean) 	
						
						local count = `count'+1
						
						//
						forv g = `fp1'/7 {
							local var : word `g' of `varlist'
							local fe7 "`fe6' `var'"
							di "`fe7'"
							qui: reghdfe log_price, absorb(`fe7') residuals(pres`count') 
							qui: sum log_price if e(sample)
							qui: replace pres`count' = pres`count' + r(mean) 	
							
							local count = `count'+1
						}
					}
				}
			}
		}
	}
}


// Again but interacted with time
cap drop mn
cap drop log_price_samp
local count=1
local varlist "bedrooms#year bathrooms#year floorarea_50#year yearbuilt_50#year condition#year heatingtype#year parking#year"
forv a = 1/7 {
	local ap1 = `a'+1
	local var : word `a' of `varlist'
	local fe1 "`var'"
	di "`fe1'"
	qui: reghdfe log_price, absorb(`fe1') residuals(tpres`count') 
	cap drop log_price_samp mn
	qui: gen log_price_samp = log_price if e(sample)
	qui: gegen mn = mean(log_price_samp), by(year)
	qui: replace tpres`count' = tpres`count' + mn 
	
	local count = `count'+1
	
	//
	forv b = `ap1'/7 {
		local bp1 = `b'+1
		local var : word `b' of `varlist'
		local fe2 "`fe1' `var'"
		di "`fe2'"
		qui: reghdfe log_price, absorb(`fe2') residuals(tpres`count') 
		cap drop log_price_samp mn
		qui: gen log_price_samp = log_price if e(sample)
		qui: gegen mn = mean(log_price_samp), by(year)
		qui: replace tpres`count' = tpres`count' + mn 
		
		local count = `count'+1
		
		//
		forv c = `bp1'/7 {
			local cp1 = `c'+1
			local var : word `c' of `varlist'
			local fe3 "`fe2' `var'"
			di "`fe3'"
			qui: reghdfe log_price, absorb(`fe3') residuals(tpres`count') 
			cap drop log_price_samp mn
			qui: gen log_price_samp = log_price if e(sample)
			qui: gegen mn = mean(log_price_samp), by(year)
			qui: replace tpres`count' = tpres`count' + mn 	
			
			local count = `count'+1
			
			//
			forv d = `cp1'/7 {
				local dp1 = `d'+1
				local var : word `d' of `varlist'
				local fe4 "`fe3' `var'"
				di "`fe4'"
				qui: reghdfe log_price, absorb(`fe4') residuals(tpres`count') 
				cap drop log_price_samp mn
				qui: gen log_price_samp = log_price if e(sample)
				qui: gegen mn = mean(log_price_samp), by(year)
				qui: replace tpres`count' = tpres`count' + mn 				
	
				local count = `count'+1
				
				//
				forv e = `dp1'/7 {
					local ep1 = `e'+1
					local var : word `e' of `varlist'
					local fe5 "`fe4' `var'"
					di "`fe5'"
					qui: reghdfe log_price, absorb(`fe5') residuals(tpres`count') 
					cap drop log_price_samp mn
					qui: gen log_price_samp = log_price if e(sample)
					qui: gegen mn = mean(log_price_samp), by(year)
					qui: replace tpres`count' = tpres`count' + mn 
					
					local count = `count'+1
					
					//
					forv f = `ep1'/7 {
						local fp1 = `f'+1 
						local var : word `f' of `varlist'
						local fe6 "`fe5' `var'"
						di "`fe6'"
						qui: reghdfe log_price, absorb(`fe6') residuals(tpres`count') 
						cap drop log_price_samp mn
						qui: gen log_price_samp = log_price if e(sample)
						qui: gegen mn = mean(log_price_samp), by(year)
						qui: replace tpres`count' = tpres`count' + mn 
						
						local count = `count'+1
						
						//
						forv g = `fp1'/7 {
							local var : word `g' of `varlist'
							local fe7 "`fe6' `var'"
							di "`fe7'"
							qui: reghdfe log_price, absorb(`fe7') residuals(tpres`count') 
							cap drop log_price_samp mn
							qui: gen log_price_samp = log_price if e(sample)
							qui: gegen mn = mean(log_price_samp), by(year)
							qui: replace tpres`count' = tpres`count' + mn 
													
							local count = `count'+1
						}
					}
				}
			}
		}
	}
}

save "$working/merged_with_hedonics.dta", replace
use "$working/merged_with_hedonics.dta", clear

**************************************
* Merge other useful data
**************************************
merge m:1 year month using  "$clean/uk_interest_rates.dta", nogen keep(master match) keepusing(uk1y uk5y uk10y uk25y uk30y uk10y20 uk10y15 uk10y15_real uk10y20_real)
merge m:1 postcode using "$raw/geography/lpa_codes.dta", nogen keep(master match)
gegen lpa_code_n = group(lpa_code)

// Merge in lat/lon
rename latitude lat_rm 
rename longitude lon_rm
merge m:1 postcode using "$raw/geography/ukpostcodes.dta", nogen keep(match)

destring lat_rm lon_rm, replace
replace latitude = lat_rm if !missing(lat_rm) & abs(latitude - lat_rm)<0.01 & abs(longitude - lon_rm)<0.01
replace longitude = lon_rm if !missing(lon_rm) & abs(latitude - lat_rm)<0.01 & abs(longitude - lon_rm)<0.01

**************************************
* Take differences
**************************************
gsort property_id date_trans
foreach var of varlist	date* ///
						year quarter month ///
						price log_price* pres* tpres* ///
						rent log_rent* $hedonics_rm ///
						duration number_years tenure ///
						uk* closed_lease {
	qui by property_id: gen L_`var' = `var'[_n-1]
	cap gen d_`var' = `var' - L_`var'
}

format date* L_date* %tdNN-DD-CCYY

rename d_date_trans days_held 
gen years_held = days_held/365.25
gen years_held_n = round(years_held)
drop d_year d_quarter d_month d_number_years* d_date* d_duration* d_closed_lease

*****************************************
* Identify extensions 
*****************************************
gen extension = (duration - L_duration + years_held > 5) & !missing(L_duration) & !missing(duration)
replace extension_amount =  duration - L_duration + years_held if extension

gen not_valid_ext = (extension_amount <= 30 | (duration - extension_amount > 150) | (L_date_expired - date_trans > 365 & L_date_expired!=.)) & extension
replace extension = 0 if not_valid_ext
replace extension_amount = . if not_valid_ext

by property_id: gen L_extension = extension[_n-1]

egen has_extension = total(extension), by(property_id)
gen multiple_extensions = has_extension > 1
						
* Maturity bins
gen short_lease = leasehold & duration <= 100 & !extension
gen med_lease 	= leasehold & duration > 100 & duration <= 300 & !extension
gen long_lease 	= leasehold & duration > 300 

* Extension date
gen date_extended = date_registered
replace date_extended = L_date_expired if date_registered > date_trans & L_date_expired < date_trans
replace date_extended = date_trans if date_extended > date_trans
replace date_extended = date_expired if has_been_extended & !missing(date_expired)
format date_extended %tdDD-NN-CCYY

**********************************************************
* Drop impossible values
drop if L_duration<0 | duration<0 | extension_amount < 0
gen whb_duration = duration
replace whb_duration = L_duration - years_held if extension 

* Create more useful variables 
foreach var of varlist duration L_duration whb_duration {
	gen `var'5yr = round(`var', 5)
	gen `var'10yr = round(`var', 10)
	gen `var'20yr = round(`var', 20)
	gen `var'p1000 = `var'/1000
}

*Label variables
label var years_held "Number of Years Held"
label var year "Sale Year"
label var type "Property Type"
label var property_id "Property ID"
label var price "Sale Price"
label var L_price "Purchase Price"
label var duration "Lease Duration at Sale Time"
label var L_duration "Lease Duration at Purchase Time"
label var date_trans "Sale Date"
label var L_date_trans "Purchase Date"
label var date_registered "Registration Date"
label var date_from "Lease Origination Date"
label var date_expired "Closed Lease Expiration Date"
label var extension_amount "Extension Amount"
label var new "New Sale Dummy"
label var years_elapsed "Time Since Lease Start"
label var floorarea "Floor Area"
label var yearbuilt "Year Built"
label var log_rent "Log(Rental Price)"
label var bathrooms "Bathrooms"
label var bedrooms "Bedrooms"
label var livingrooms "Living Rooms"
label var age "Property Age"

// Get number of transactions by property 
bys property_id: gen num_trans=_N

// Would-have-been duration in 2023 (without extension)
gen duration2023 = number_years - datediff(date_from, date("January 1, 2023", "MDY"), "year") 
replace duration2023 = duration2023 - extension_amount if extension
replace duration2023 = round(duration2023)

compress
drop agent_name property_number address? name

save "$clean/flats.dta", replace

use  "$clean/flats.dta", clear
keep if leasehold
save "$clean/leasehold_flats_full.dta", replace

// Create a light-weight version of the data set
drop pres* tpres* L_pres* L_tpres* d_pres? d_pres?? d_pres??? d_tpres? d_tpres?? d_tpres???
save "$clean/leasehold_flats.dta", replace
use "$clean/leasehold_flats_full.dta", clear

// Drop lease with very short initial terms, since these are probably commercial leases and we don't want them as controls
drop if number_years < 50
sum duration2023 if extension
keep if duration2023 <= r(max)+5 | has_extension

// Drop cases that look like extensions but do not satisfy criteria 
drop if not_valid_ext 

// Drop if missing pre-transaction, since this is for repeat sales
drop if missing(L_date_trans)

keep property_id date_trans L_date_trans duration* L_duration* whb_duration years_held year L_year quarter L_quarter month L_month log_price* L_log_price* d_log_price* pres_bedrooms pres_all L_pres_all pres_linear L_pres_linear d_pres_all d_pres_linear area outcode postcode extension* latitude longitude number_years date_from date_registered extension_amount num_trans has_been_extended has_extension lpa_code region d_pres* d_tpres*

compress
export delimited "$working/for_rsi_large.csv", replace

drop d_pres? d_pres?? d_pres??? d_tpres*
export delimited "$working/for_rsi.csv", replace
