*** Create data:

*************************************
* Estimate Stability
*************************************

import delimited "$working/rsi_variations.csv", clear
tempfile rsi_variations
save `rsi_variations'

use "$clean/experiments.dta", clear 
merge 1:1 property_id date_trans using "$clean/leasehold_flats_full.dta", keep(match) nogen force

cap drop date L_date 
drop did_rsi_linear

gen date = L_year*4 + L_quarter 
sdecode area, replace 
gen duration2023_10yr = round(duration2023, 10)

merge m:1 area duration2023_10yr date using `rsi_variations', nogen keep(master match)
rename rsi* L_rsi*

replace date = year*4 + quarter
merge m:1 area duration2023_10yr date using `rsi_variations', nogen keep(master match)

foreach var of varlist rsi* {
	local tag = subinstr("`var'", "rsi_", "", .)
	di "`tag'"
	
	qui: gen d_`var' = `var' - L_`var'
	qui: gen did_`var' = `tag' - d_`var'
}

keep if year>=2000
gen ystar = .
gen controls = ""
gen time_interaction = .

local count = 1
foreach var of varlist did* {
		
	if strpos("`var'", "tpres") {
		local tag = subinstr("`var'", "d_tpres", "", .)
		local interact = 1
		
	} 
	else {
		local tag = subinstr("`var'", "d_pres", "", .)
		local interact = 0
	}
	local tag = subinstr("`tag'", "did_rsi_", "", .)
	local tag = subinstr("`tag'", "_", "", .)
	if "`var'"=="did_rsi" local tag = "none"
	
	di "`tag'"
	qui: nl (`var' = ln(1-exp(-({ystar=3}/100)*(T+k))) - ln(1-exp(-({ystar=3}/100)*T)))
	
	qui: replace ystar = _b[/ystar] if _n==`count'
	qui: replace controls = "`tag'" if _n==`count'
	qui: replace time_interaction = `interact' if _n==`count'
	
	local count = `count'+1
}

keep ystar controls time_interaction
drop if missing(ystar)

local varlist "bedrooms bathrooms floorarea_50 age_50 condition_n heating_n parking_n"

foreach var of local varlist {
	gen `var' = 0
}

local count=1
forv a = 1/7 {
	local ap1 = `a'+1
	local var : word `a' of `varlist'
	local fe1 "`var'"
	di "`count': `fe1'"
	foreach v of local fe1 { 
		qui: replace `v' = 1 if controls=="`count'"
	}
	
	local count = `count'+1
	
	//
	forv b = `ap1'/7 {
		local bp1 = `b'+1
		local var : word `b' of `varlist'
		local fe2 "`fe1' `var'"
		di "`count':`fe2'"
		foreach v of local fe2 { 
			qui: replace `v' = 1 if controls=="`count'"
		}
		
		local count = `count'+1
		
		//
		forv c = `bp1'/7 {
			local cp1 = `c'+1
			local var : word `c' of `varlist'
			local fe3 "`fe2' `var'"
			di "`count':`fe3'"
			foreach v of local fe3 { 
				qui: replace `v' = 1 if controls=="`count'"
			}
			
			local count = `count'+1
			
			//
			forv d = `cp1'/7 {
				local dp1 = `d'+1
				local var : word `d' of `varlist'
				local fe4 "`fe3' `var'"
				di "`count':`fe4'"
				foreach v of local fe4 { 
					qui: replace `v' = 1 if controls=="`count'"
				}
				
				local count = `count'+1
				
				//
				forv e = `dp1'/7 {
					local ep1 = `e'+1
					local var : word `e' of `varlist'
					local fe5 "`fe4' `var'"
					di "`count':`fe5'"
					foreach v of local fe5 { 
						qui: replace `v' = 1 if controls=="`count'"
					}					
					
					local count = `count'+1
					
					//
					forv f = `ep1'/7 {
						local fp1 = `f'+1 
						local var : word `f' of `varlist'
						local fe6 "`fe5' `var'"
						di "`count':`fe6'"
						foreach v of local fe6 { 
							qui: replace `v' = 1 if controls=="`count'"
						}
						
						local count = `count'+1
						
						//
						forv g = `fp1'/7 {
							local var : word `g' of `varlist'
							local fe7 "`fe6' `var'"
							di "`count':`fe7'"
							foreach v of local fe7 { 
								qui: replace `v' = 1 if controls=="`count'"
							}
							
							local count = `count'+1
						}
					}
				}
			}
		}
	}
}


save "$clean/quasi_experimental_stability.dta", replace

********************************************
* Cross-Sectional Stability
********************************************

use "$clean/flats.dta", clear

// Drop freeholds that switch between leasehold and freehold, because these may be transactions of the underlying freehold
gegen tot_freehold=total(freehold), by(property_id)
bys property_id: gen N=_N
drop if freehold & tot_freehold!=N

// Use same sample as in our data
keep if year>=2000

// For each group, get the mean freehold price
gegen g=group(outcode year quarter)
gegen pct_fh = mean(freehold), by(g)
drop if pct_fh==0 | pct_fh==1

foreach var of varlist log_price pres* tpres* {
	di "`var'"
	qui: gen `var'_fh = `var' if freehold
	qui: gegen `var'_fh = mean(`var'_fh), by(g) replace
	qui: gen `var'_discount = `var' - `var'_fh if leasehold
}	

* Calculate rate of return for all variations of hedonics
gen ystar = .
gen controls = ""
gen time_interaction = .

local count = 130
foreach var of varlist pres*discount tpres*discount log_price_discount {
	
	if strpos("`var'", "tpres") {
		local tag = subinstr("`var'", "tpres", "", .)
		local interact = 1
		
	} 
	else {
		local tag = subinstr("`var'", "pres", "", .)
		local interact = 0
	}
	local tag = subinstr("`tag'", "_discount", "", .)
	local tag = subinstr("`tag'", "_", "", .)
	
	di "`tag'"

	qui: sum `var'
	if r(N)==0 continue
	cap nl ( `var' = ln(1-exp(-({ystar}/100) * duration)) ), initial(ystar 4) variables(`var' duration) vce(robust)
	
	if _rc==0 {
		if "`var'"=="log_price_discount" local tag = "none"
		
		qui: replace ystar = _b[/ystar] if _n==`count'
		qui: replace controls = "`tag'" if _n==`count'
		qui: replace time_interaction = `interact' if _n==`count'
	}
	local count = `count'+1
}

keep ystar controls time_interaction
drop if missing(ystar)
save "$clean/cross_sectional_stability.dta", replace
