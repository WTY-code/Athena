echo "Set variable CDTHOME: ${CDTHOME}"

if [ ${CDTIP} ];then
	echo "Set DNS's ip: ${CDTIP}"
else
    # Auto-detect IP if not set
    export CDTIP=$(hostname -I | awk '{print $1}')
	echo -e "\033[33m CDTIP IS NOT SET. Using detected IP: ${CDTIP} \033[0m"
    if [ -z "${CDTIP}" ]; then
        echo -e "\033[31m Failed to detect IP. Please export CDTIP manually. \033[0m"
        exit 1
    fi
fi


docker rm cdt

docker run -v  ${CDTHOME}:/hyperledger/caliper/workspace \
--dns ${CDTIP} --name=cdt \
cdt caliper launch master \
--caliper-workspace  /hyperledger/caliper/workspace \
--caliper-benchconfig dist/config-distributed.yaml \
--caliper-networkconfig dist/client.yaml


echo "Test finish and benchmark result will be saved into ${CDTHOME}/report.html."