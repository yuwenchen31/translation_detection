#!/bin/bash

# MTLD and HD-D are measures for lexical diversity that do not suffer the drawbacks of TTR

MTLD="python3 ./mtld_and_hdd.py"
YULES="python3 ../third/yulesK.py"
YULES2="python3 ../third/lexical_diversity_yule.py"
DTRX=../datasets/taraxu/taraxu-set/
DMS=../datasets/MS/
DTED=../datasets/wit3/


function mtld {
	F=$1

	BF=$(basename $F)
	RES=$($MTLD $F)
	echo -e "MTLD $BF\t$RES"
}


function yules {
	F=$1

	BF=$(basename $F)
	#RES=$($YULES < $F)
	#echo -e "YULES $BF\t$RES"

	RES=$($YULES2 $F)
	echo -e "YULES2 $BF\t$RES"
}


function mtlds {
	# ----- TRX -----
	echo "TTR trx en-de"
	for F in $DTRX/*ref.en-de.norm.tok $DTRX/*edit.en-de.norm.tok $DTRX/*hyp.en-de.norm.tok; do
		mtld $F
		yules $F
	done

#	exit

	echo "TTR trx de-en"
	for F in $DTRX/*ref.de-en.norm.tok $DTRX/*edit.de-en.norm.tok $DTRX/*hyp.de-en.norm.tok; do
		mtld $F
		yules $F
	done

	echo "TTR trx es-de"
	for F in $DTRX/*ref.es-de.norm.tok $DTRX/*edit.es-de.norm.tok $DTRX/*hyp.es-de.norm.tok; do
		mtld $F
		yules $F
	done

	echo "TTR trx de-es"
	for F in $DTRX/*ref.de-es.norm.tok $DTRX/*edit.de-es.norm.tok $DTRX/*hyp.de-es.norm.tok; do
		mtld $F
		yules $F
	done



	# ----- MS -----
	echo "TTR ms zh-en"
	for F in $DMS/orig.*en.norm.tok; do
		mtld $F
		yules $F
	done

	# ----- IWSLT -----
	echo "TTR ted en-de"
	for F in $DTED/*de.norm.tok; do
		mtld $F
		yules $F
	done

	echo "TTR ted en-fr"
	for F in $DTED/*fr.norm.tok; do
		mtld $F
		yules $F
	done
}

mtlds


