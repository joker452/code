module ChurchNumerals (
c0, cSucc, cToInt, cPlus, cMult, cExp)where

c0 = \f -> \x -> x

cSucc cx f = cx f . f

cToInt cx = cx (+1) 0

cPlus cm cn f = cm f . cn f

cMult cm cn = cm . cn

cExp cm cn = cn cm



