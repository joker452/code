module ChurchNumerals (
c0, cSucc, cToInt, cPlus, cMult, cExp)where

c0 = \f -> \x -> x

cSucc cx f = cx f . f

cToInt cx = cx (+1) 0

cPlus cm cn f = cm f . (cn f)

cMultHelper m cn res = if m > 1 then cMultHelper (m - 1) cn (cPlus cn res)
                       else cPlus cn res
cMult cm cn = cMultHelper m cn c0 where m = cToInt cm

cExp cm cn = cn cm



