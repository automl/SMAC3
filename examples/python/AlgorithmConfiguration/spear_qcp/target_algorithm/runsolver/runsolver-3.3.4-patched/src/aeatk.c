/*
 * aeatk.c
 *
 *  Created on: May 7, 2014
 *      Author: sjr
 */

#include <arpa/inet.h>
#include <netinet/in.h>

#include <sys/types.h>
#include <sys/socket.h>


/**
 * This function is called by the timerThread at the end of updating runsolvers internal
 * time measure and will send a message to the port on the local machine.
 *
 * Written by Stephen Ramage
 */
void writeCPUTimeToSocket(double cpuTime) {

	static char* portstr = getenv("AEATK_PORT");

	static int noMessageDisplay = 0;

	if (portstr == NULL) {
		if (noMessageDisplay == 0) {
			noMessageDisplay++;
			printf(
					"[AEATK] No environment variable \"AEATK_PORT\" detected not sending updates\n\n");
		}
		return;
	}

	//Safely convert port to int
	int port = strtol(portstr, NULL, 10);

	if (port < 1024 || port > 65535) {
		//Invalid port, probably nothing

		if (noMessageDisplay == 0) {
			noMessageDisplay++;
			printf(
					"[AEATK] Invalid port set in \"AEATK_PORT\" must be in [1024, 65535]\n\n");
		}
		return;

	}

	static struct sockaddr_in servaddr;

	static double updateFreqDbl = 1;
	static int lastUpdate = -1;

	static int sockfd = -1;

	static char* updateFreq = getenv("AEATK_CPU_TIME_FREQUENCY");

	if (sockfd == -1) {
		sockfd = socket(AF_INET, SOCK_DGRAM, 0);

		bzero(&servaddr, sizeof(servaddr));
		servaddr.sin_family = AF_INET;
		servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
		servaddr.sin_port = htons(port);

	}

	if (updateFreq != NULL) {
		//Runsolver won't actually call this function faster than every 2 seconds
		//But we set it to 1 second anyway
		updateFreqDbl = strtod(updateFreq, NULL);

		if (updateFreqDbl < 1) {
			updateFreqDbl = 1;
		}
	}

	if ((time(NULL) - lastUpdate) > updateFreqDbl) {
		lastUpdate = time(NULL);

		static char buf[100];

		int length = sprintf(buf, "%f\n", cpuTime);

		//Not sure if MSG_DONTWAIT flag is appropriate but it didn't seem to matter when I tried a few times

		if (noMessageDisplay >= 2) {
			noMessageDisplay++;
			printf(
					"[AEATK] Sending CPUTime updates to 127.0.0.1 on port %d every %.3f seconds, last time %.3f \n\n",
					port, updateFreqDbl, cpuTime);
		}

		sendto(sockfd, buf, length, 0, (struct sockaddr*) &servaddr,
				sizeof(servaddr));
		return;
	}

	return;
}

void updateCores() {

	char* set_affinity = getenv("AEATK_SET_TASK_AFFINITY");

	if ((set_affinity == NULL) || strcmp(set_affinity, "1") != 0) {
		printf(
				"[AEATK] No environment variable \"AEATK_SET_TASK_AFFINITY\" detected, cores will be treated normally\n\n");
		return;

	}

	char* taskstr = getenv("AEATK_CONCURRENT_TASK_ID");

	if (taskstr == NULL) {
		printf(
				"[AEATK] No environment variable \"AEATK_CONCURRENT_TASK_ID\" detected, cores are treated normally\n\n");
		return;

	}

	unsigned int taskid = strtoul(taskstr, NULL, 10);

	printf(
			"[AEATK] This version of runsolver restricts subprocesses to only one core when \"AEATK_CONCURRENT_TASK_ID\" is set.\n");

	vector<unsigned short int> cores2;

	getAllocatedCoresByProcessorOrder(cores2);

	if (taskid >= cores2.size()) {
		cout << "[AEATK] taskid: " << taskid
				<< " is greater than the number of cores we have available cores: "
				<< cores2.size() << " affinity: ";

		printAllocatedCores(cout, cores2);

		cout << " something is wrong, exiting" << endl;
		exit(1);
	}

	vector<unsigned short int> cores;
	cores.push_back(cores2[taskid]);

	cpu_set_t mask = affinityMask(cores);
	if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) != 0) {
		perror("sched_setaffinity failed: ");
		cout << "[AEATK] Couldn't set affinity " << endl;
		exit(1);
	}

	return;
}


