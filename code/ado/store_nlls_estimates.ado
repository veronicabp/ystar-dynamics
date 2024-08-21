
program define store_nlls_estimates
    version 16
	matrix b = e(b)
	matrix V = e(V)

	matrix b = (b)
	matrix V = (V)

	matrix coleq b = " "
	matrix coleq V = " "

	matrix colnames b = "`1'"
	matrix colnames V = "`1'"

	erepost b = b V = V, rename
	estimates store `2'
end
