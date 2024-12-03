
program define rsi
    version 16
	cap drop outcome_var
	cap drop date_var 
	cap drop L_date_var
	cap drop rsi L_rsi d_rsi
	
	// Get arguments
	qui: gen outcome_var = `1'
    qui: gen date_var = `2'
	qui: gen L_date_var = L_`2'
	if "`3'" != "" local cond "`3'"
	else local cond = 1
	
	******** Calculate repeat sales index ********
	gsort date_var
	
	sum date_var 
	local date_min = r(min)
	di `date_min'
	
	qui: levelsof date_var, local(dates)
	foreach date of local dates {
		if round(`date',10)==`date' di `date'
		if `date'==`date_min' continue
		qui: gen DATE`date' = 0
		qui: replace DATE`date' = 1 if date_var==`date'
		qui: replace DATE`date' = -1 if L_date_var==`date'
	}
	
	reg outcome_var DATE* if `cond' & date_var!=L_date_var, vce(robust)
	predict d_rsi 
	replace d_rsi = _b[_cons] if date_var==L_date_var
	drop outcome_var date_var L_date_var DATE*
end
