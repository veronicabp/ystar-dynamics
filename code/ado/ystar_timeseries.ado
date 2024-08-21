
program define ystar_timeseries
    version 16
	cap drop lhs_var 
	cap drop sample
	cap drop weight
	
	// Get arguments
    qui: gen lhs_var = `1'
	qui: if "`2'" != "" gen sample = `2'
	qui: else gen sample = 1
	local tag "`3'"
	
	qui: if "`4'" != "" gen weight = `4'
	qui: else gen weight = 1
	
	******** Calculate y* ********
	cap drop xaxis
	qui: gen xaxis = _n+1999 if _n+1999<=2023
	
	// Pre-2003
	qui: nl (lhs_var = ln(1-exp(-({ystar=3}/100)*(T+k))) - ln(1-exp(-({ystar=3}/100)*T))) if year<=2003 & sample [aw=weight]
	qui: gen ystar`tag'=_b[/ystar] if xaxis<=2003 & _b[/ystar]<=20 & _se[/ystar]!=.
	qui: gen ub`tag'= _b[/ystar] + 1.96*_se[/ystar] if xaxis<=2003 & _b[/ystar]<=20 & _se[/ystar]!=.
	qui: gen lb`tag'= _b[/ystar] - 1.96*_se[/ystar] if xaxis<=2003 & _b[/ystar]<=20 & _se[/ystar]!=.
 
	// Post-2003 yearly
	forv year=2004/2023 {
		di `year'
		qui: nl (lhs_var = ln(1-exp(-({ystar=3}/100)*(T+k))) - ln(1-exp(-({ystar=3}/100)*T))) if year==`year' & sample [aw=weight]
		qui: replace ystar`tag' = _b[/ystar] if xaxis==`year' & _b[/ystar]<=20 & _se[/ystar]!=.
		qui: replace ub`tag' = _b[/ystar] + 1.96*_se[/ystar] if xaxis==`year' & _b[/ystar]<=20 & _se[/ystar]!=.
		qui: replace lb`tag' = _b[/ystar] - 1.96*_se[/ystar] if xaxis==`year' & _b[/ystar]<=20 & _se[/ystar]!=.
	}
	
	drop weight
	drop sample
end
