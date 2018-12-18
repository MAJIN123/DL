每段保留一个空行：awk 'BEGIN{b=0} {if($0==""){b=1;next;} if(b){print "\n"$0;b=0;}else print;}' input >output
