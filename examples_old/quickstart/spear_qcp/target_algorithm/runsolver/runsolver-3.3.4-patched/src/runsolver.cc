/*
 * Copyright (C) 2010 Olivier ROUSSEL
 *
 * This file is part of runsolver.
 *
 * runsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * runsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with runsolver.  If not, see <http://www.gnu.org/licenses/>.
 */

// if TIMESTAMPSEPARATESTDOUTANDERR is defined, when time stamping the
// solver output, stdout and stderr are transmitted separately to
// runsolver. This lets us print whether the line comes from stdout or
// stderr. Unfortunately, when streams are separated, we have no
// guarantee to get the lines in the order they were sent (try with
// testtimestamper.cc). For this reason, this flag should be
// undefined. When undefined, stdout and stderr are combined together
// in the solver and are transmitted through one single pipe to
// runsolver. Anyway, this is the usual way to pipe a program output
// to another program input
#undef TIMESTAMPSEPARATESTDOUTANDERR

#define SENDSIGNALBOTTOMUP

/*
 * TODO
 *
 * - arrange so that display occurs each n*period seconds (and not n*(period+epsilon))
 * - print the command line of new processes (??)
 *
 * - use /proc/%pid/status
 *
 * - man pthreads : le comportement de la mesure du temps pour les
 *   threads depend de la version du noyau
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/times.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>

#include <sys/ipc.h>
#include <sys/msg.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <getopt.h>

#include "Cores.hh"
#include "SignalNames.hh"
#include "ProcessList.hh"
#include "CircularBufferFilter.hh"

#ifdef WATCHSYSCALLS
#include "SyscallsTracer.hh"
#endif

#ifndef SVNVERSION
#define SVNVERSION
#endif

#ifndef VERSION
#define VERSION
#endif

#include "TimeStamper.hh"
#include "ProcessTree.hh"
#include "ProcessHistory.hh"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <sys/types.h>
#include <sys/socket.h>

#include "aeatk.c"

using namespace std;

const char *version=VERSION;

#define SIGWAIT_RETURN_ERRORCODE 25
#define PTHREAD_SIGMASK_ERROR_CODE 26
#define PTHREAD_CREATE_ERROR_CODE 27





/**
 * Flag variable which indicates whether the solver is presently running.
 */
bool solverIsRunningFlag;

pthread_mutex_t solverIsRunningFlagLock = PTHREAD_MUTEX_INITIALIZER;
void setSolverIsRunning(bool running)
{
	pthread_mutex_lock(&solverIsRunningFlagLock);
	solverIsRunningFlag = running;
	pthread_mutex_unlock(&solverIsRunningFlagLock);
}
bool getSolverIsRunning()
{
	bool returnValue;
	pthread_mutex_lock(&solverIsRunningFlagLock);
	returnValue = solverIsRunningFlag;
	pthread_mutex_unlock(&solverIsRunningFlagLock);
	return returnValue;
}



/**
 * Controls the timer thread, and once set true it signals
 * that the timer thread should shutdown gracefully.
 */
bool timerThreadShouldShutdown;

pthread_mutex_t timerThreadShouldShutdownFlagLock = PTHREAD_MUTEX_INITIALIZER;
void setTimerThreadShouldShutdown(bool shouldShutdown)
{
	pthread_mutex_lock(&timerThreadShouldShutdownFlagLock);
	timerThreadShouldShutdown = shouldShutdown;
	pthread_mutex_unlock(&timerThreadShouldShutdownFlagLock);
}
bool getTimerThreadShouldShutdownFlag()
{
	bool returnValue;
	pthread_mutex_lock(&timerThreadShouldShutdownFlagLock);
	returnValue = timerThreadShouldShutdown;
	pthread_mutex_unlock(&timerThreadShouldShutdownFlagLock);
	return returnValue;
}


/**
 *  This is essentially a CAS that replaces the old
 *  logic with respect to stopSolver(). If multiple threads
 *  call stopSolver() only one of them will be allowed to pass, the
 *  others will return immediately.
 */

bool stopSolverInvokedByCallerFlag;
pthread_mutex_t stopSolverFlagLock = PTHREAD_MUTEX_INITIALIZER;
void initStopSolverInvokedByCallerFlag()
{
	pthread_mutex_lock(&stopSolverFlagLock);
	stopSolverInvokedByCallerFlag = false;
	pthread_mutex_unlock(&stopSolverFlagLock);
}


bool compareAndSetStopSolverInvokedByCallerFlag(bool expect, bool newValue)
{
	bool changed;
	pthread_mutex_lock(&stopSolverFlagLock);
	if(stopSolverInvokedByCallerFlag == expect)
	{
		stopSolverInvokedByCallerFlag = newValue;
		changed = true;
	} else
	{
		changed = false;
	}
	pthread_mutex_unlock(&stopSolverFlagLock);
	return changed;
}


/**
 *  This condition variable is set, once
 *  the stopSolver() method has finished being called,
 *  it essentially releases the other thread to start executing.
 */
pthread_mutex_t stopSolverCalledLock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t stopSolverCalledCondition = PTHREAD_COND_INITIALIZER;


bool stopSolverShouldExecuteFlag;
void initStopSolverShouldExecuteFlag()
{
	pthread_mutex_lock(&stopSolverCalledLock);
	stopSolverShouldExecuteFlag = false;
	pthread_mutex_unlock(&stopSolverCalledLock);
}

void setStopSolverShouldExecuteFlag()
{
	pthread_mutex_lock(&stopSolverCalledLock);
	stopSolverShouldExecuteFlag = true;
	pthread_cond_broadcast(&stopSolverCalledCondition);
	pthread_mutex_unlock(&stopSolverCalledLock);
}

void waitForStopSolverShouldExecuteFlag()
{

	pthread_mutex_lock(&stopSolverCalledLock);

	while(1)
	{
		if(stopSolverShouldExecuteFlag == true)
		{
			pthread_mutex_unlock(&stopSolverCalledLock);
			return;
		} else
		{
			pthread_cond_wait(&stopSolverCalledCondition, &stopSolverCalledLock);
		}
	}

	return;
}




pthread_mutex_t timerThreadStoppedLock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t timerThreadStoppedCondition = PTHREAD_COND_INITIALIZER;


/**
 * This flag should be set true after the timer thread
 * has finished. According to the pthreads_join() main page,
 * multiple threads cannot call pthread_join() and so
 * we have falled back to the main thread creating the thread,
 * joining, and then other threads that want to know can watch this condition
 */
bool timerThreadStopped;
void initTimerThreadStopped()
{
        pthread_mutex_lock(&timerThreadStoppedLock);
        timerThreadStopped = false;
        pthread_mutex_unlock(&timerThreadStoppedLock);
}

void setTimerThreadStopped()
{
        pthread_mutex_lock(&timerThreadStoppedLock);
        timerThreadStopped = true;
        pthread_cond_broadcast(&timerThreadStoppedCondition);
        pthread_mutex_unlock(&timerThreadStoppedLock);
}

void waitForTimerThreadStopped()
{

        pthread_mutex_lock(&timerThreadStoppedLock);

        while(1)
        {
                if(timerThreadStopped == true)
                {
                        pthread_mutex_unlock(&timerThreadStoppedLock);
                        return;
                } else
                {
                        pthread_cond_wait(&timerThreadStoppedCondition, &timerThreadStoppedLock);
                }
        }

        return;
}



 /**
 * use only one instance
 *
 */
class RunSolver {
private:
	class Limit {
	private:
		int resource;
		rlim_t limit;
	protected:
		const char *name, *unit; // name and unit of this limit
		int scale; // call setrlimit() with limit<<scale. Useful to
				   // maintain a limit in KiB while setrlimit() needs a
				   // limit in bytes (set scale=10)
	public:
		/**
		 * resource==-1 means that this is a fake limit (which is not
		 * enforced via setrlimit())
		 */
		Limit(int resource) :
				resource(resource) {
			scale = 0;
		}

		void setLimit(rlim_t lim) {
			limit = lim;
		}

		void output(ostream &out) {
			out << "Enforcing " << name << ": " << limit << " " << unit << endl;
		}

		void outputEnforcedLimit(ostream &out) {
			out << "Current " << name << ": " << getEnforcedLimit() << " "
					<< unit << endl;
		}

		void enforceLimit() {
			if (resource < 0)
				return; // this is a fake limit, don't enforce anything

			struct rlimit lim = { limit, limit };

			if (getrlimit(resource, &lim)) {
				perror("getrlimit failed");
				return;
			}

			lim.rlim_cur = limit << scale;
			if (setrlimit(resource, &lim))
				perror("setrlimit failed");
		}

		rlim_t getEnforcedLimit() {
			if (resource < 0)
				return limit; // this is a fake limit, don't ask the system

			struct rlimit lim;

			if (getrlimit(resource, &lim)) {
				perror("getrlimit failed");
				return (rlim_t) -1; // good value to return ???
			}

			return lim.rlim_cur >> scale;
		}
	};

public:
	/**
	 * This is a fake limit. It doesn't enforce anything by its own.
	 */
	class SoftVSIZELimit: public Limit {
	public:
		SoftVSIZELimit(rlim_t size) :
				Limit(-1) {
			name = "VSIZE limit (soft limit, will send SIGTERM then SIGKILL)";
			unit = "KiB";
			scale = 10;
			setLimit(size);
		}
	};

	class HardVSIZELimit: public Limit {
	public:
		HardVSIZELimit(rlim_t size) :
				Limit(RLIMIT_AS) {
			name =
					"VSIZE limit (hard limit, stack expansion will fail with SIGSEGV, brk() and mmap() will return ENOMEM)";
			unit = "KiB";
			scale = 10;
			setLimit(size);
		}
	};

	/**
	 * This is a fake limit. It doesn't enforce anything by its own.
	 */
	class SoftCPULimit: public Limit {
	public:
		SoftCPULimit(rlim_t cpuTime) :
				Limit(-1) {
			name = "CPUTime limit (soft limit, will send SIGTERM then SIGKILL)";
			unit = "seconds";
			setLimit(cpuTime);
		}
	};

	/**
	 * This is a fake limit. It doesn't enforce anything by its own.
	 */
	class WallClockLimit: public Limit {
	public:
		WallClockLimit(rlim_t wallClockTime) :
				Limit(-1) {
			name =
					"wall clock limit (soft limit, will send SIGTERM then SIGKILL)";
			unit = "seconds";
			setLimit(wallClockTime);
		}
	};

	class HardCPULimit: public Limit {
	public:
		HardCPULimit(rlim_t cpuTime) :
				Limit(RLIMIT_CPU) {
			name = "CPUTime limit (hard limit, will send SIGXCPU)";
			unit = "seconds";
			setLimit(cpuTime);
		}
	};

	class FileSizeLimit: public Limit {
	public:
		FileSizeLimit(rlim_t fileSize) :
				Limit(RLIMIT_FSIZE) {
			name = "FSIZE limit";
			unit = "KiB";
			scale = 10;
			setLimit(fileSize);
		}
	};

	class NumberOfFilesLimit: public Limit {
	public:
		NumberOfFilesLimit(rlim_t nbFiles) :
				Limit(RLIMIT_NOFILE) {
			name = "NbFILES limit";
			unit = "files";
			setLimit(nbFiles);
		}
	};

	class StackSizeLimit: public Limit {
	public:
		/**
		 * won't enforce limit
		 */
		StackSizeLimit() :
				Limit(RLIMIT_STACK) {
			name = "StackSize limit";
			unit = "KiB";
			scale = 10;
		}

		/**
		 * will enforce limit
		 */
		StackSizeLimit(rlim_t size) :
				Limit(RLIMIT_STACK) {
			name = "Stack size limit";
			unit = "KiB";
			scale = 10;
			setLimit(size);
		}
	};

	static const unsigned long int clockTicksPerSecond;

private:
	// when the solver uses less than cpuUsageThreshold % of the CPU,
	// try to identify process of other users which use more than
	// heavyProcessThreshold % of the CPU
	static const float cpuUsageThreshold = 0.8; // % of CPU
	static const float heavyProcessThreshold = 0.1; // % of CPU

	pid_t childpid; // pid of the process we're watching
	//Improperly synchronized so no longer used: bool solverIsRunning;

	ProcessTree *procTree, *lastProcTree; // to gather data about a process and its children
	ProcessHistory procHistory; // history of the last procTree

#ifdef WATCHSYSCALLS
	// list of processes we're watching
	ProcessList childrenList;
#endif

	// list of all tasks created by the solver. Only updated where
	// cleanupSolverOwnIPCQueues is set
	set<pid_t> listAllProcesses;

	float limitCPUTime; // CPU time accorded to solver before we stop it
	float limitWallClockTime; // Wall clock time given to solver before we stop it
	long limitVSize; // VSize accorded to solver before we stop it

#ifdef WATCHSYSCALLS
	bool InterceptKernelCalls; // should we intercept kernel system calls ?
#endif

private:
	static RunSolver *instance; // instance of RunSolver for signal handlers

	clock_t start, stop;

	// used in timerThread
	float maxDisplayPeriod; // max. period between two samplings

	struct timeval starttv, stoptv;
	float elapsed; // elapsed time in seconds since the start of the
				   // watched process. Updated by timerThread
	float lastDisplayedElapsedTime; // last elapsed time at which a
	// process tree was displayed
	struct tms tmp;

#ifdef WATCHSYSCALLS
	class MemOverQuotaCallback : public SyscallCallback
	{
	public:
		MemOverQuotaCallback() {alreadyActivated=false;}

		virtual void operator() ()
		{
			if (!alreadyActivated)
			{
				alreadyActivated=true;
				RunSolver::instance->stopSolver("Mem limit exceeded: sending SIGTERM then SIGKILL");
			}
		}
	private:
		bool alreadyActivated;
	};
#endif

	/**
	 * signal handler to gather data about the solver
	 *
	 * may be called for SIGINT, SIGTERM or SIGALRM
	 */
/*	static void watcherSigHandler(int signum, siginfo_t *siginfo,
			void *ucontext) {

		cout << ""
#ifdef debug
		cout << "signal handler called for " << getSignalName(signum) << endl;
#endif

#ifdef disabled
		// if we don't use the SA_NOCLDSTOP flag for sigaction() when
		// registering the handler for SIGCHLD, we get a SIGCHLD each time
		// the traced process is stopped by a sigtrap. These must be
		// ignored. The code below ignores all SIGCHLD including the one
		// which let us know that the process we're watching has
		// terminated (despite the fact that this is something we want to
		// know). For this reason, this code is disabled and we use
		// SA_NOCLDSTOP

		if (signum==SIGCHLD)
		return;
#endif

		if (signum == SIGTERM || signum == SIGINT) {
			instance->stopSolver("Received SIGTERM or SIGINT, killing child");
			return;
		}
	}
*/


	/**
	 * This function will handle signal requests synchronously.
	 */
	static void * synchronousSignalHandler(void *arg)
	{

		int err, signo;

		sigset_t mask;


		sigemptyset(&mask);
		getSignalMask(mask);

		while(1)
		{
			err = sigwait(&mask, &signo);


#ifdef debug
			cout << "signal handler called for " << getSignalName(signum) << endl;
#endif

			if( err != 0 )
			{
				cout << "Error occurred while calling sigwait";
				exit(SIGWAIT_RETURN_ERRORCODE);
			}

#ifdef disabled
			// if we don't use the SA_NOCLDSTOP flag for sigaction() when
			// registering the handler for SIGCHLD, we get a SIGCHLD each time
			// the traced process is stopped by a sigtrap. These must be
			// ignored. The code below ignores all SIGCHLD including the one
			// which let us know that the process we're watching has
			// terminated (despite the fact that this is something we want to
			// know). For this reason, this code is disabled and we use
			// SA_NOCLDSTOP

			if (signo==SIGCHLD)
			continue;
#endif
			switch(signo)
			{
				case SIGINT:
				case SIGTERM:
					cout << "Recieved signal " << getSignalName(signo) << " setting flag.";
					setTimerThreadShouldShutdown(true);
					instance->stopSolver("Received SIGTERM or SIGINT, killing child");
					return NULL;
					/*if (signum == SIGTERM || signum == SIGINT) {
								instance->stopSolver("Received SIGTERM or SIGINT, killing child");
								return;
					}*/
			}


		}
		return NULL;

	}



	/**
	 * procedure run by the thread in charge of gathering data about the
	 * watched process
	 */
	static void *timerThread(void *) {
		struct timespec d, delay = { 0, 100000000 }; // 0.1 s
		struct timeval tv;
		float displayPeriod, nextDisplayTime = 0;
		int count = 0;

		displayPeriod = 0.1;

		while (getSolverIsRunning())
		{

			d = delay;

			// try to compensate possible delays
			d.tv_nsec -= ((tv.tv_usec - instance->starttv.tv_usec + 1000000)
					% 100000) * 1000;



			//We nano sleep but poll for shutdown.
			// wait (use a loop in case of an interrupt)


			while (nanosleep(&d, &d) == -1 && errno == EINTR)
			{
				//This is on purpose;
			}
			if(getTimerThreadShouldShutdownFlag())
			{
				break;
			}



			// get currenttime
			gettimeofday(&tv, NULL);
			instance->elapsed = tv.tv_sec + tv.tv_usec / 1E6
					- instance->starttv.tv_sec
					- instance->starttv.tv_usec / 1E6;

			if (count == 10) {
				// identify new children
				if (instance->readProcessData(true)) {
					setSolverIsRunning(false);
					cout << "Root process has terminated, timer thread aborting";
					break;
				}
				count = 0;
			} else {
				// simple update
				if (instance->readProcessData(false)) {
					setSolverIsRunning(false);
					cout << "Root process has terminated, timer thread aborting";
					break;
				}
				count++;
			}

			if (instance->elapsed >= nextDisplayTime) {

				instance->displayProcessData();
				nextDisplayTime += displayPeriod;
				displayPeriod = std::min(2 * displayPeriod,
						instance->maxDisplayPeriod);
				//cout << "displayPeriod=" << displayPeriod << endl;
				//cout << "nextDisplayTime=" << nextDisplayTime << endl;
			}
		}

		if(getTimerThreadShouldShutdownFlag())
		{
			cout << "Timer Thread detected shutdown flag, shutting down";

		}

		setTimerThreadStopped();
		return NULL; // meaningless
	}

	/**
	 * gather data about the watched processes
	 *
	 * if updateChildrenList is false, only update process informations
	 * but don't try to identify new processes (faster), if
	 * updateChildrenList is true, identify all processes which are a
	 * children of a watched process.
	 *
	 * @return true iff the main process was terminated
	 */
	bool readProcessData(bool updateChildrenList = true) {
		if (!childpid)
			return false; // trying to collect data before parent got the pid

#ifdef debug
		cout << "Reading process data" << endl;
#endif

		lastProcTree = procTree;
		procTree = new ProcessTree(*procTree);

		procTree->setElapsedTime(elapsed);

		if (updateChildrenList) {
			procTree->readProcesses();
			if (cleanupSolverOwnIPCQueues)
				procTree->listProcesses(listAllProcesses);
		} else
			procTree->updateProcessesData();

		if (procTree->rootProcessEnded()) {
			delete procTree;
			procTree = lastProcTree;
			return true; // don't go any further
		}

		lastCPUTime = currentCPUTime;
		lastSystemTime = currentSystemTime;
		lastUserTime = currentUserTime;
		lastVSize = currentVSize;

		procTree->currentCPUTime(currentUserTime, currentSystemTime);
		currentCPUTime = currentUserTime + currentSystemTime;

		currentVSize = procTree->currentVSize();

		maxVSize = max(maxVSize, currentVSize);

		if (currentCPUTime < lastCPUTime) {
			lostCPUTime += lastCPUTime - currentCPUTime;
			lostSystemTime += lastSystemTime - currentSystemTime;
			lostUserTime += lastUserTime - currentUserTime;

			cout << "\n############\n# WARNING:\n"
					<< "# current cumulated CPU time (" << currentCPUTime
					<< " s) is less than in the last sample (" << lastCPUTime
					<< " s)\n"
					<< "# The time of a child was probably not reported to its father.\n"
					<< "# (see the two samples below)\n"
					<< "# Adding the difference ("
					<< lastCPUTime - currentCPUTime
					<< " s) to the 'lost time'.\n";

			if (lastProcTree) {
				lastProcTree->dumpProcessTree(cout);
				lastProcTree->dumpCPUTimeAndVSize(cout, lastCPUTime, lastVSize);
			}

			procTree->dumpProcessTree(cout);
			procTree->dumpCPUTimeAndVSize(cout, currentCPUTime, currentVSize);

			cout << "#\n############\n" << endl;
		}

		procHistory.push(procTree);

		tStamp.setCPUtimeFromAnotherThread(currentCPUTime + lostCPUTime);

		if (limitCPUTime && currentCPUTime + lostCPUTime >= limitCPUTime) {
			cout << "runsolver_max_cpu_time_exceeded" << endl;
			stopSolver(	"Maximum CPU time exceeded: sending SIGTERM then SIGKILL");
		}

		if (limitWallClockTime && elapsed >= limitWallClockTime) {
			stopSolver("Maximum wall clock time exceeded: sending SIGTERM then SIGKILL");
		}

		if (limitVSize && currentVSize >= limitVSize) {
			cout << "runsolver_max_memory_limit_exceeded" << endl;
			stopSolver("Maximum VSize exceeded: sending SIGTERM then SIGKILL");
		}

		writeCPUTimeToSocket(currentCPUTime);

		return false;
	}

	/**
	 * display the data we have about all watched processes
	 *
	 */
	void displayProcessData() {
		lastDisplayedElapsedTime = procTree->getElapsedTime();
		procTree->dumpProcessTree(cout);
		procTree->dumpCPUTimeAndVSize(cout, currentCPUTime, currentVSize);

		if (elapsed > 2 && currentCPUTime < cpuUsageThreshold * elapsed) {
			// its looks like we're not using all CPU. Maybe there's another
			// heavy process running. Try to identify it.
			procTree->dumpHeavyProcesses(cout, heavyProcessThreshold);
		}
	}

	/**
	 * Install the signal handlers
	 */
	void installSignalHandler() {


		int err;
		sigset_t mask;
		sigset_t oldmask;
		pthread_t tid;
		getSignalMask(mask);
		err = pthread_sigmask(SIG_BLOCK, &mask, &oldmask);
		if(err != 0)
		{
			cout << "[ERROR] Couldn't block signals, error occurred exiting:" << err << endl << flush;
			exit(PTHREAD_SIGMASK_ERROR_CODE);
		}

		err = pthread_create(&tid, NULL, synchronousSignalHandler, 0);

		if (err != 0)
		{
			cout << "[ERROR] Couldn't create signal handling thread, exiting:" << err <<  endl << flush;
			exit(PTHREAD_CREATE_ERROR_CODE);
		}

		//sigaddset(&mask);


		/**
		 * This was the old code for it, but it is NOT
		 * safe for a multi-threaded environment.
		 * See Advanced Programming in Unix Environment 2nd Edition p415
		 * (Threads and Signals)
		 *
		 *
		 *
		struct sigaction handler;
		handler.sa_sigaction = watcherSigHandler;
		sigemptyset(&handler.sa_mask);
		handler.sa_flags = SA_SIGINFO | SA_NOCLDSTOP | SA_RESTART;

		 The SA_RESTART flag tells that system calls which are
		 interrupted by a signal should be automatically restarted. This
		 way, we don't have to encapsulate system calls in a loop which
		 would restart them when they return with
		 errno=EINTR. Alternatively, we could have used
		 siginterrupt().

		 The SA_NOCLDSTOP prevent us from getting a SIGCHLD each time a
		 process is stopped for tracing.
		 *//*
		sigaction(SIGALRM, &handler, NULL);
		sigaction(SIGINT, &handler, NULL);
		sigaction(SIGTERM, &handler, NULL);
		*/
	}

	static void getSignalMask(sigset_t& mask) {
		sigemptyset(&mask);
		sigaddset(&mask, SIGALRM);
		sigaddset(&mask, SIGINT);
		sigaddset(&mask, SIGTERM);
	}

	vector<Limit *> limits;

	ofstream out;

	// when set, redirect the standard input of the child to this file
	char *inputRedirectionFilename;

	// when set, redirect the standard output of the child to this file
	char *outputRedirectionFilename;

	// when set, save the most relevant information in an easy to parse
	// format consisting of VAR=VALUE lines
	char *varOutputFilename;

	// a class to timestamp each line of some output streams
	static TimeStamper tStamp;

	bool timeStamping; // true iff the solver output must be timestamped
	int stdoutRedirFD[2]; // pipe for timestamping the solver stdout
#ifdef TIMESTAMPSEPARATESTDOUTANDERR
	int stderrRedirFD[2]; // pipe for timestamping the solver stderr
#endif
	bool usePTY; // use a PTY to collect output from the solver ? (to
	// force the solver to line-buffer its output)
	int ptymaster; // PTY master

	bool cleanupSolverOwnIPCQueues; // delete IPC queues that the solver may have created?
	bool cleanupAllIPCQueues; // delete all IPC queues that are owned by the user on exit

	pthread_t timeStamperTID; // tid of the thread which does the timestamping
	//Moved to run() method
    //Moved to run() method
	//Removed no longer needed.

	// limits imposed on the size of the solver output (0 if unlimited)
	unsigned long long int limitedOutputActivateSize, limitedOutputMaxSize;

	CircularBufferFilter limitedOutputSizeFilter; // a filter to limit
	// the solver output
	// size

	// time in seconds between a SIGTERM and a SIGKILL sent to children
	int delayBeforeKill;

	// maximum cumulated vsize of all children
	long maxVSize;

	// current CPU time of the watched process
	float currentCPUTime, currentSystemTime, currentUserTime;

	// last CPU time obtained from the watcher
	float lastCPUTime, lastUserTime, lastSystemTime;

	// sometimes, the time of a child is not reported to its father
	// (e.g. when the father didn't wait its child). We try to detect
	// this and cumulate the lost time in these variables
	float lostCPUTime, lostUserTime, lostSystemTime;

	long currentVSize; // current VSize of the watched process
	long lastVSize; // last VSize of the watched process

	streambuf *coutSaveBuf; // backup of the cout buf (in case of a redirection)

public:
	RunSolver() :
			procHistory(10) {


		maxDisplayPeriod = 60;

		limitCPUTime = 0; // no CPU limit by default
		limitWallClockTime = 0; // no CPU limit by default
		limitVSize = 0; // no memory limit by default

		delayBeforeKill = 2;

		inputRedirectionFilename = NULL;
		outputRedirectionFilename = NULL;
		varOutputFilename = NULL;
		timeStamping = false;

		maxVSize = 0;



#ifdef WATCHSYSCALLS
		InterceptKernelCalls=false;
#endif

		coutSaveBuf = NULL;

		currentCPUTime = 0;
		currentUserTime = 0;
		currentSystemTime = 0;

		lastCPUTime = 0;
		lastUserTime = 0;
		lastSystemTime = 0;

		lostCPUTime = 0;
		lostUserTime = 0;
		lostSystemTime = 0;

		limitedOutputActivateSize = 0;
		limitedOutputMaxSize = 0;

		currentVSize = 0;
		lastVSize = 0;

		usePTY = false;
		cleanupAllIPCQueues = false;
		cleanupSolverOwnIPCQueues = false;

		lastProcTree = NULL;
		procTree = new ProcessTree();
		procHistory.push(procTree);


	}

	~RunSolver() {
		// cancel redirection before we leave
		if (coutSaveBuf)
			cout.rdbuf(coutSaveBuf);

		// procTree and lastProcTree are deleted by ~ProcessHistory()
	}

	/**
	 * use a PTY to collect the solver output
	 */
	void setUsePTY(bool usePTY) {
		this->usePTY = usePTY;
	}

	/**
	 * delete IPC queues that the solver may have created
	 */
	void setCleanupSolverOwnIPCQueues(bool cleanup) {
		cleanupSolverOwnIPCQueues = cleanup;
	}

	/**
	 * delete all IPC queues that were created by this user
	 */
	void setCleanupAllIPCQueues(bool cleanup) {
		cleanupAllIPCQueues = cleanup;
	}

	/**
	 * send the output of the watching process to a given file
	 */
	void setWatcherOutputFile(char *filename) {
		out.open(filename);
		coutSaveBuf = cout.rdbuf(out.rdbuf());
	}

	/**
	 * send the output of the watching process to a given file
	 */
	void setVarOutputFile(char *filename) {
		varOutputFilename = filename;
	}

	/**
	 * redirect the standard input of the solver to a given file
	 */
	void setInputRedirection(char *filename) {
		inputRedirectionFilename = filename;
	}

	/**
	 * redirect the standard output of the solver to a given file
	 */
	void setOutputRedirection(char *filename) {
		outputRedirectionFilename = filename;
	}

	/**
	 * decide if we should timestamp the solver output or not
	 */
	void setTimeStamping(bool val) {
		timeStamping = val;
	}

	/**
	 * decide if we should add an EOF line to the solver output or not
	 */
	void setTimeStampingAddEOF(bool val) {
		if (!timeStamping)
			throw runtime_error(
					"EOF line can only be added when timestamping is on");

		tStamp.addEOF(val);
	}

	/**
	 * limit the size of the solver output
	 */
	void setSolverOutputLimits(unsigned long long int activateSize,
			unsigned long long int maxSize) {
		if (!timeStamping)
			throw runtime_error(
					"limit on the output size can only be enforced when timestamping is on");

		limitedOutputActivateSize = activateSize;
		limitedOutputMaxSize = maxSize;
	}

	/**
	 * set the time we should wait between sending a SIGTERM and a
	 * SIGKILL to a solver we want to stop
	 */
	void setDelayBeforeKill(int seconds) {
		delayBeforeKill = seconds;
	}

	void setCPULimit(int sec) {
		limitCPUTime = sec;
		// SoftCPULimit doesn't enforce anything by its own
		addLimit(new SoftCPULimit(sec));

		addLimit(new HardCPULimit(sec + 30));
		// add an extra delay because we want to stop the solver by
		// stopSolver() instead of SIGXCPU
	}

	void setWallClockLimit(int sec) {
		limitWallClockTime = sec;
		// WallClockLimit doesn't enforce anything by its own
		addLimit(new WallClockLimit(sec));
	}

	/**
	 * limits are expressed in kilobytes
	 *
	 * soft limit= limit (calls stopSolver())
	 * hard limit= limit+reserve (causes immediate SIGKILL)
	 */
	void setMemLimit(long limit, long reserve) {
#ifdef WATCHSYSCALLS
		if (InterceptKernelCalls)
		{
			addLimit(new SoftVSIZELimit((limit)));

			heapSizeWatcher.setMaximumCumulatedSize(limit);

			// add an extra amount of memory because we want to stop the solver by
			// stopSolver() instead of SIGKILL
			addLimit(new HardVSIZELimit((limit+reserve)));
		}
		else
#endif
		{
			limitVSize = limit;
			// SoftVSIZELimit doesn't enforce anything by its own
			addLimit(new SoftVSIZELimit((limit)));

			// add an extra amount of memory because we want to stop the solver by
			// stopSolver() instead of SIGKILL
			addLimit(new HardVSIZELimit((limit + reserve)));
		}
	}

	/**
	 * ask to intercept system calls to enforce the policy concerning
	 * file and network accesses
	 *
	 * slows down the solver
	 */
	void watchSyscalls() {
#ifdef WATCHSYSCALLS
		cout << "watching syscalls " << endl;
		InterceptKernelCalls=true;
#else
		throw runtime_error("disabled function");
#endif
	}

	/**
	 * add a limit to respect
	 *
	 * @parm limit: must be dynamically allocated
	 */
	void addLimit(Limit *limit) {
		limits.push_back(limit);
	}

	void printLimits(ostream &s) {
		for (size_t i = 0; i < limits.size(); ++i)
			limits[i]->output(s);

		if (limitedOutputMaxSize) {
			if (timeStamping)
				cout << "Solver output will be limited to a maximum of "
						<< limitedOutputMaxSize << " bytes. The first "
						<< limitedOutputActivateSize << " bytes and the last "
						<< limitedOutputMaxSize - limitedOutputActivateSize
						<< " bytes will be preserved" << endl;
			else
				cout << "Solver output limit is ignored (currently only "
						"available with timestamping)" << endl;
		}

		if (timeStamping && usePTY)
			cout << "Using a pseudo terminal to collect output from the solver"
					<< endl;
	}

	/**
	 * select cores that will be available to the process we watch
	 *
	 * if physicalView is true, desc contains the id of the cores to be
	 * used as they are known on the system. In this case, cores 0 and
	 * 1 (for example) may belong to different processors.
	 *
	 * if physicalView is false, desc contains the index of the cores to
	 * be used in a list of available cores sorted by the processor to
	 * which they belong. In this case, cores 0 and 1 (for example) will
	 * necessarily belong to the same processor (unless it has only one
	 * core!)
	 */
	void selectCores(const string &desc, bool physicalView) {
		vector<unsigned short int> availableCores, selectedCores;

		getExistingCores(availableCores, physicalView);

		istringstream d(desc);
		size_t a, b;

		while (true) {
			if (!(d >> a)) {
				printCoresListSyntax();
				exit(1);
			}

			while (isspace(d.peek()))
				d.get();

			if (d.peek() == '-') {
				d.get();
				if (d >> b) {
					//cout << "read " << a << "-" << b << endl;
					if (b < a)
						swap(a, b);

					for (size_t i = a; i <= b; ++i)
						if (i >= 0 && i < availableCores.size())
							selectedCores.push_back(availableCores[i]);
				} else {
					printCoresListSyntax();
					exit(1);
				}
			} else {
				//cout << "read " << a << endl;
				selectedCores.push_back(availableCores[a]);
			}

			if (d.peek() == ',')
				d.get();
			else if (d.peek() == EOF)
				break;
			else {
				printCoresListSyntax();
				exit(1);
			}
		}

		cpu_set_t mask = affinityMask(selectedCores);
		if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) != 0)
			perror("sched_setaffinity failed: ");
	}

	void printCoresListSyntax() {
		cout << "Syntax of a core list:\n"
				<< "  range first-last or individual numbers separated by commas\n"
				<< "  examples: 0-1,5,7 or 0,1,5,7 or 0-7\n" << endl;
	}

#ifdef WATCHSYSCALLS
	// for system calls watcher
	SyscallsTracer syscallsTracer;

	FileWatcher fileWatcher;
	NetworkWatcher networkWatcher;

	//MemOverQuotaCallback memOverQuotaCallback;
	//HeapSizeWatcher heapSizeWatcher;
#endif

	void initSignalHandlers()
	{

	}

	/**
	 * run a solver
	 */
	void run(char **cmd) {
		instance = this;

		installSignalHandler();
		initStopSolverInvokedByCallerFlag();
		initTimerThreadStopped();
		initStopSolverShouldExecuteFlag();
		setTimerThreadShouldShutdown(false);
		setSolverIsRunning(false);
		
		childpid = 0;

#ifdef WATCHSYSCALLS
		if (InterceptKernelCalls)
		{
			syscallsTracer.registerAction(&fileWatcher);
			syscallsTracer.registerAction(&networkWatcher);

			//heapSizeWatcher.setOverQuotaHandler(&memOverQuotaCallback);
			//syscallsTracer.registerAction(&heapSizeWatcher);

			// intercept system calls that create and destroy processes and
			// maintain a list of subprocesses
			syscallsTracer.registerAction(new ProcessWatcher(childrenList));
		}
#endif

		if (timeStamping) {
			int fd = STDOUT_FILENO;

			if (outputRedirectionFilename) {
				fd = open(outputRedirectionFilename,
						O_WRONLY | O_CREAT | O_TRUNC, 0644);
				if (fd < 0)
					throw runtime_error(
							string("open failed during output redirection: ")
									+ strerror(errno));
			}

			int outputFromSolverFD = 0;

			if (usePTY) {
				ptymaster = posix_openpt(O_RDWR);
				if (ptymaster < 0) {
					perror("Failed to create pseudo-terminal");
					exit(1);
				}

				outputFromSolverFD = ptymaster;

				if (grantpt(ptymaster) != 0) {
					perror("Failed to grant the pseudo-terminal");
					exit(1);
				}

				if (unlockpt(ptymaster) != 0) {
					perror("Failed to unlock the pseudo-terminal");
					exit(1);
				}
			} else {
				pipe(stdoutRedirFD);
#ifdef TIMESTAMPSEPARATESTDOUTANDERR
				pipe(stderrRedirFD);
#endif
			}

#ifdef TIMESTAMPSEPARATESTDOUTANDERR
			tStamp.watch(stdoutRedirFD[0],'o',fd); // 'o' as output
			tStamp.watch(stderrRedirFD[0],'e',fd);// 'e' as error
#else

			if (usePTY)
				outputFromSolverFD = ptymaster;
			else
				outputFromSolverFD = stdoutRedirFD[0];

			if (limitedOutputMaxSize) {
				limitedOutputSizeFilter.setup(fd, limitedOutputActivateSize,
						limitedOutputMaxSize);
				tStamp.watch(outputFromSolverFD, &limitedOutputSizeFilter, 0);
			} else
				tStamp.watch(outputFromSolverFD, 0, fd);
#endif
			tStamp.resetTimeStamp();

			int err = pthread_create(&timeStamperTID, NULL, timeStampThread,
					NULL);
			if (err)
				cout
						<< "Failed to create a thread to timestamp the solver output"
						<< endl;
		}

		start = times(&tmp);
		gettimeofday(&starttv, NULL);
		childpid = fork();
		if (childpid < 0) {
			perror("fork failed");
			exit(127);
		} else if (childpid == 0) {
			// child

			// enforce limits
			for (size_t i = 0; i < limits.size(); ++i)
				limits[i]->enforceLimit();

			StackSizeLimit stackLimit;
			stackLimit.outputEnforcedLimit(cout);
			cout << endl;

			// create a new process group (for several reasons, see for
			// example ProcessTree::rootProcessEnded())
			setpgid(0, 0);

#ifdef WATCHSYSCALLS
			if (InterceptKernelCalls)
			syscallsTracer.childTraceKernelCalls();
#endif

			if (inputRedirectionFilename) {
				int err;
				int fd;

				fd = open(inputRedirectionFilename, O_RDONLY);
				if (fd < 0)
					throw runtime_error(
							string("open failed during input redirection: ")
									+ strerror(errno));

				err = dup2(fd, STDIN_FILENO);
				if (err < 0)
					throw runtime_error(
							string("dup2 failed during input redirection: ")
									+ strerror(errno));

				close(fd);
			}

			if (outputRedirectionFilename && !timeStamping) {
				int err;
				int fd;

				fd = open(outputRedirectionFilename,
						O_WRONLY | O_CREAT | O_TRUNC, 0644);
				if (fd < 0)
					throw runtime_error(
							string("open failed during output redirection: ")
									+ strerror(errno));

				err = dup2(fd, STDOUT_FILENO);
				if (err < 0)
					throw runtime_error(
							string("dup2 failed during output redirection: ")
									+ strerror(errno));

				err = dup2(fd, STDERR_FILENO);
				if (err < 0)
					throw runtime_error(
							string("dup2 failed during output redirection: ")
									+ strerror(errno));

				close(fd);
			}

			if (timeStamping) {
				if (usePTY) {
					int err;
					int fd;

					char *pts = ptsname(ptymaster);

					if (pts == NULL)
						throw runtime_error(
								string("Failed to get pty slave name:")
										+ strerror(errno));

					fd = open(pts, O_RDWR);
					if (fd < 0)
						throw runtime_error(
								string("open of pty slave failed: ")
										+ strerror(errno));

					err = dup2(fd, STDOUT_FILENO);
					if (err < 0)
						throw runtime_error(
								string(
										"dup2 failed during output redirection: ")
										+ strerror(errno));

					err = dup2(fd, STDERR_FILENO);
					if (err < 0)
						throw runtime_error(
								string(
										"dup2 failed during output redirection: ")
										+ strerror(errno));

					close(fd);
				} else {
					// plain tube

					// redirecting stdout and stderr to the write side of the
					// pipes to runsolver; close the read side of the pipe which
					// belongs to our father

					close(stdoutRedirFD[0]);
					dup2(stdoutRedirFD[1], STDOUT_FILENO);
					close(stdoutRedirFD[1]);

#ifdef TIMESTAMPSEPARATESTDOUTANDERR
					close(stderrRedirFD[0]);
					dup2(stderrRedirFD[1],STDERR_FILENO);
					close(stderrRedirFD[1]);
#else
					dup2(STDOUT_FILENO, STDERR_FILENO);
#endif
				}
			}

			// ??? check the way it uses PATH
			execvp(cmd[0], cmd);
			// only returns when it failed
			perror("exec failed");

			int i = 0;
			cerr << "Solver command line was: ";
			while (cmd[i])
				cerr << cmd[i++] << ' ';

			cerr << '\n' << endl;

			exit(127);
		} else {
			// parent

			setSolverIsRunning(true);

			// We don't care about stdin. In case someone writes 'echo
			// data | runsolver program', runsolver just closes its stdin
			// and the child will be the only one to access it.
			close(STDIN_FILENO);

			// let runsolver run on the last allocated core
			vector<unsigned short int> cores;

			getAllocatedCoresByProcessorOrder(cores);
			if (cores.size() > 1) {
				int tmp = cores.back();
				cores.clear();
				cores.push_back(tmp);
			}

			cpu_set_t mask = affinityMask(cores);
			if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) != 0)
				perror("sched_setaffinity failed: ");

#ifdef debug
			cout << "child has pid " << childpid << endl;
#endif

#ifdef WATCHSYSCALLS
			// add the child to the list of processes we're watching
			childrenList.add(childpid);

			if (InterceptKernelCalls)
			syscallsTracer.parentTraceKernelCalls(childpid);
#endif

			procTree->setDefaultRootPID(childpid);

			if (timeStamping && !usePTY) {
				// the write side of the pipe belongs to the child
				close(stdoutRedirFD[1]);
#ifdef TIMESTAMPSEPARATESTDOUTANDERR
				close(stderrRedirFD[1]);
#endif
			}

			procTree->setElapsedTime(0);
			procTree->readProcesses();
			if (cleanupSolverOwnIPCQueues)
				procTree->listProcesses(listAllProcesses);
			procTree->dumpProcessTree(cout);

			pthread_t timerThreadTID; // tid of the thread which watches the process
			int err = pthread_create(&timerThreadTID, NULL, timerThread, NULL);

			if (err)
				cout << "Failed to create the timer thread" << endl;

			pthread_t stopSolverThreadTID; // tid of the thread which kills the process

			//We create the stop solver thread now,
			//It will wait for some condition variables before actually executing.
			err = pthread_create(&stopSolverThreadTID, NULL, waitAndKillSolver,NULL); 
			if (err)
				 cout << "Failed to create a thread to stop solver" << endl;


			int childstatus;
			struct rusage childrusage;
			int wait4result;

#ifdef WATCHSYSCALLS
			if (InterceptKernelCalls)
			syscallsTracer.watchKernelCalls(childpid,childstatus,childrusage);
			else
#endif
			wait4result = wait4(childpid, &childstatus, 0, &childrusage);

			//This will dispatch the stop solver thread
			instance->stopSolver("Solver termination detected (return from wait4)");

			//This should cause the timer thread to terminate
			
			cout << "Requesting Timer Thread Should Shutdown \n";
			setTimerThreadShouldShutdown(true);

			cout << "Setting solver is running = false \n";
			instance->stop = times(&instance->tmp);
			gettimeofday(&instance->stoptv, NULL);

			setSolverIsRunning(false);


			//Wait for the timer thread to terminate
			cout << "Joining on timer Thread \n";
			pthread_join(timerThreadTID, NULL);

			cout << "Timer Thread Done\n";
		

			
			// the program we started has just terminated but some of its
			// processes can still be running. Kill them all!
			// A kill(-childpid,SIGKILL) alone is not sufficient because
			// children may have created their own session
			/*
			if (procTree) {
				procTree->sendSignalNow(SIGKILL);
				// just in case a new process was created in between
				procTree->sendSignalToProcessGroups(SIGKILL);
			} else {
				// fallback
				
			}*/

			if (timeStamping) {
				// wait for the time stamper thread to output the last lines
				pthread_join(timeStamperTID, NULL);
			}

			cout << "Waiting for stop solver thread\n";
			// wait for the stop solver thread to output the last lines
			pthread_join(stopSolverThreadTID, NULL);

			cout << "Sending fallback SIGKILL signal to process group\n";

			//We no longer do the procees tree approach, instead at this point
			//we have already done our best attempt at a clean up.
			//So we will just always call kill just in case.
			kill(-childpid, SIGKILL);
			

			cout << endl;

			cout << "Solver just ended. Dumping a history of the "
					"last processes samples" << endl;

			procHistory.dumpHistory(cout, lastDisplayedElapsedTime);
			cout << endl;

			if (WIFEXITED(childstatus))
				cout << "Child status: " << WEXITSTATUS(childstatus) << endl;
			else if (WIFSIGNALED(childstatus)) {
				int sig = WTERMSIG(childstatus);

				cout << "Child ended because it received signal " << sig << " ("
						<< getSignalName(sig) << ")" << endl;

#ifdef WCOREDUMP
				if (WCOREDUMP(childstatus))
					cout << "Child dumped core" << endl;
#endif

			} else if (WIFSTOPPED(childstatus)) {
				cout << "Child was stopped by signal "
						<< getSignalName(WSTOPSIG(childstatus)) << endl;
			} else {
				cout << "Child ended for unknown reason !!" << endl;
			}

			float realtime; // Elapsed real milliseconds
			float virtualtime; // Elapsed virtual (CPU) milliseconds
			float virtualUserTime; // Elapsed virtual (User CPU) milliseconds
			float virtualSystemTime; // Elapsed virtual (System CPU) milliseconds

			//realtime=(stop-start)*1000/(float)clockTicksPerSecond;

			realtime = stoptv.tv_sec * 1000 + stoptv.tv_usec / 1000.0
					- starttv.tv_sec * 1000 - starttv.tv_usec / 1000.0;

#if 1
			// use wait4 data
			virtualtime = childrusage.ru_utime.tv_sec * 1000
					+ childrusage.ru_utime.tv_usec / 1000.0
					+ childrusage.ru_stime.tv_sec * 1000
					+ childrusage.ru_stime.tv_usec / 1000.0;

			virtualUserTime = childrusage.ru_utime.tv_sec * 1000
					+ childrusage.ru_utime.tv_usec / 1000.0;

			virtualSystemTime = childrusage.ru_stime.tv_sec * 1000
					+ childrusage.ru_stime.tv_usec / 1000.0;

			// don't get fooled
			if (currentCPUTime > virtualtime / 1000
					|| virtualtime / 1000 > currentCPUTime + 60) {
				cout << "\n# WARNING:\n"
						<< "# CPU time reported by wait4() is probably wrong !\n"
						<< "# wait4(...,&childrusage) returns " << wait4result;

				if (wait4result < 0)
					cout << " (errno=" << errno << ", \"" << strerror(errno)
							<< "\")";

				cout << " and gives\n" << "#  childrusage.ru_utime.tv_sec="
						<< childrusage.ru_utime.tv_sec
						<< "\n#  childrusage.ru_utime.tv_usec="
						<< childrusage.ru_utime.tv_usec
						<< "\n#  childrusage.ru_stime.tv_sec="
						<< childrusage.ru_stime.tv_sec
						<< "\n#  childrusage.ru_stime.tv_usec="
						<< childrusage.ru_stime.tv_usec << endl;

				cout << "# CPU time returned by wait4() is "
						<< virtualtime / 1000 << endl
						<< "# while last known CPU time is " << currentCPUTime
						<< "\n" << "#\n";

				if (currentCPUTime > virtualtime / 1000)
					cout
							<< "# Solver probably didn't/couldn't wait for its children"
							<< endl;

				cout << "# Using CPU time of the last sample as value...\n";
				cout << endl;

				virtualtime = currentCPUTime * 1000;
				virtualUserTime = currentUserTime * 1000;
				virtualSystemTime = currentSystemTime * 1000;
			}
#else
			// use the watcher data
			virtualtime=currentCPUTime*1000;
			virtualUserTime=currentUserTime*1000;
			virtualSystemTime=currentSystemTime*1000;
#endif

			if (lostCPUTime != 0) {
				cout << endl << "# WARNING:" << endl;
				cout
						<< "# The CPU time of some children was not reported to their father\n"
						<< "# (probably because of a missing or aborted wait())."
						<< endl
						<< "# This 'lost CPU time' is added to the watched process CPU time."
						<< endl;
				cout << "#  lost CPU time (s): " << lostCPUTime << endl;
				cout << "#  lost CPU user time (s): " << lostUserTime << endl;
				cout << "#  lost CPU system time (s): " << lostSystemTime
						<< endl << endl;

				virtualtime += lostCPUTime * 1000;
				virtualUserTime += lostUserTime * 1000;
				virtualSystemTime += lostSystemTime * 1000;
			}

			//cout << "Real time (s): " << realtime/1000.0 << endl;
			cout << "Real time (s): " << realtime / 1000.0 << endl;
			cout << "runsolver_walltime: " << realtime / 1000.0 << endl;
			cout << "CPU time (s): " << virtualtime / 1000.0 << endl;
			cout << "runsolver_cputime: " << virtualtime / 1000.0 << endl;
			cout << "CPU user time (s): " << virtualUserTime / 1000.0 << endl;
			cout << "CPU system time (s): " << virtualSystemTime / 1000.0
					<< endl;
			cout << "CPU usage (%): "
					<< 100 * virtualtime
							/ (float) ((realtime != 0) ? realtime : 1) << endl;
			cout << "Max. virtual memory (cumulated for all children) (KiB): "
					<< maxVSize << endl;

			if (cleanupAllIPCQueues || cleanupSolverOwnIPCQueues)
				cleanupIPCMsgQueues();

			struct rusage r;
			getrusage(RUSAGE_CHILDREN, &r);

			cout << endl;
			cout << "getrusage(RUSAGE_CHILDREN,...) data:" << endl;
			cout << "user time used= "
					<< r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1E-6 << endl;
			cout << "system time used= "
					<< r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1E-6 << endl;
			cout << "maximum resident set size= " << r.ru_maxrss << endl;
			cout << "integral shared memory size= " << r.ru_ixrss << endl;
			cout << "integral unshared data size= " << r.ru_idrss << endl;
			cout << "integral unshared stack size= " << r.ru_isrss << endl;
			cout << "page reclaims= " << r.ru_minflt << endl;
			cout << "page faults= " << r.ru_majflt << endl;
			cout << "swaps= " << r.ru_nswap << endl;
			cout << "block input operations= " << r.ru_inblock << endl;
			cout << "block output operations= " << r.ru_oublock << endl;
			cout << "messages sent= " << r.ru_msgsnd << endl;
			cout << "messages received= " << r.ru_msgrcv << endl;
			cout << "signals received= " << r.ru_nsignals << endl;
			cout << "voluntary context switches= " << r.ru_nvcsw << endl;
			cout << "involuntary context switches= " << r.ru_nivcsw << endl;
			cout << endl;

			if (varOutputFilename) {
				ofstream var(varOutputFilename);

				var << "# WCTIME: wall clock time in seconds\n" << "WCTIME="
						<< realtime / 1000.0 << endl;

				var << "# CPUTIME: CPU time in seconds\n" << "CPUTIME="
						<< virtualtime / 1000.0 << endl;

				var << "# USERTIME: CPU time spent in user mode in seconds\n"
						<< "USERTIME=" << virtualUserTime / 1000.0 << endl;

				var
						<< "# SYSTEMTIME: CPU time spent in system mode in seconds\n"
						<< "SYSTEMTIME=" << virtualSystemTime / 1000.0 << endl;

				var << "# CPUUSAGE: CPUTIME/WCTIME in percent\n" << "CPUUSAGE="
						<< 100 * virtualtime
								/ (float) ((realtime != 0) ? realtime : 1)
						<< endl;

				var << "# MAXVM: maximum virtual memory used in KiB\n"
						<< "MAXVM=" << maxVSize << endl;
			}

			getrusage(RUSAGE_SELF, &r);
			cout << "runsolver used "
					<< r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1E-6
					<< " second user time and "
					<< r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1E-6
					<< " second system time\n" << endl;

			cout << "The end" << endl;
		}
	}

	//bool stopSolverStarted;

	/**
	 * properly stop a solver
	 *
	 * to be used when the caller cannot wait (for example from a
	 * system call callback).
	 */
	void stopSolver(const char *msg) {

		if(!compareAndSetStopSolverInvokedByCallerFlag(false, true))
		{
			//*Stop solver was already called;
			return;
		}


		/**
		 * We want the timer thread to shutdown, prior to the stopSolverThread continuing.
		 */
		setTimerThreadShouldShutdown(true);

		/*The solver will first receive a SIGTERM to give it a chance to
		 output the best solution it found so far (in the case of an
		 optimizing solver). A few seconds later, the program will receive a
		 SIGKILL signal from the controlling program to terminate the
		 solver.*/

		cout << "\n\n\nShutdown Reason:\n" << msg << endl;


		// (Previously sent SIGTERM as soon as possible, not sure why)
		// Now we just start the other thread. The other thread won't fire until the
		//timer thread is dead.
		setStopSolverShouldExecuteFlag();
	}

	/**
	 * procedure run by the thread in charge of killing the solver
	 */
	static void *waitAndKillSolver(void *) {
		cout << "Stop Solver Thread: Started\n";
		waitForStopSolverShouldExecuteFlag();
		cout << "Stop Solver Thread: Should Execute\n";
		waitForTimerThreadStopped();
		cout << "Stop Solver Thread: Timer Thread Stopped\n";

		// give some evidence to the user
		RunSolver::instance->readProcessData();
		RunSolver::instance->displayProcessData();

		RunSolver::instance->sendSIGTERM();
		cout << "Stop Solver Thread: SIGTERM sent, waiting and sending SIGKILL\n";
		RunSolver::instance->waitAndSendSIGKILL();
		cout << "Stop Solver Thread: DONE\n";
		return NULL;
	}

	/**
	 * procedure run by the thread in charge of timestamping the solver
	 * output stream
	 */
	static void *timeStampThread(void *) {
		tStamp.timeStampLines(); // enless loop

		return NULL;
	}

	void sendSIGTERM() {
#ifdef SENDSIGNALBOTTOMUP
		cout << "\nSending SIGTERM to process tree (bottom up)" << endl;
		procTree->sendSignalBottomUp(SIGTERM);
#else
		// signal the whole group
		pid_t pidToSignal=-childpid;

		cout << "\nSending SIGTERM to " << pidToSignal << endl;
		kill(pidToSignal,SIGTERM);
#endif
	}

	void waitAndSendSIGKILL() {
		struct timespec delay = { delayBeforeKill, 0 };

		cout << "Sleeping " << delayBeforeKill << " seconds" << endl;

		// use a loop in case of an interrupt
		while (getSolverIsRunning() && nanosleep(&delay, &delay) == -1
				&& errno == EINTR)
			;

		if (!getSolverIsRunning())
			return;

#ifdef SENDSIGNALBOTTOMUP
		cout << "\nSending SIGKILL to process tree (bottom up)" << endl;
		procTree->sendSignalBottomUp(SIGKILL);
#endif

		// signal the whole group, in case we missed a process
		pid_t pidToSignal = -childpid;

		cout << "\nSending SIGKILL to " << pidToSignal << endl;
		kill(pidToSignal, SIGKILL);
	}

	/**
	 * delete IPC queues that may have been created by the solver
	 */
	void cleanupIPCMsgQueues() {
#ifndef __linux__
#error This code is linux specific
#endif

		struct msginfo msginfo;
		struct msqid_ds msgqueue;
		int maxid, msqid;
		uid_t myUid = geteuid();

		cout << "\n";

		maxid = msgctl(0, MSG_INFO,
				reinterpret_cast<struct msqid_ds *>(&msginfo));
		if (maxid < 0)
			return;

		for (int id = 0; id <= maxid; ++id) {
			msqid = msgctl(id, MSG_STAT, &msgqueue);

			if (msqid < 0)
				continue;

			if (msgqueue.msg_perm.cuid == myUid
					&& (cleanupAllIPCQueues
							|| listAllProcesses.find(msgqueue.msg_lspid)
									!= listAllProcesses.end()
							|| listAllProcesses.find(msgqueue.msg_lrpid)
									!= listAllProcesses.end())) {
				cout << "deleting IPC queue " << msqid;
				if (msgctl(msqid, IPC_RMID, &msgqueue))
					cout << " (failed: " << strerror(errno) << ")";
				cout << "\n";
			}
		}
	}
};

// static members
RunSolver *RunSolver::instance = NULL;
const unsigned long int RunSolver::clockTicksPerSecond = sysconf(_SC_CLK_TCK);
TimeStamper RunSolver::tStamp;

static struct option longopts[] = {
		{ "cpu-limit", required_argument, NULL, 'C' }, { "wall-clock-limit",
				required_argument, NULL, 'W' }, { "mem-limit",
				required_argument, NULL, 'M' }, { "stack-limit",
				required_argument, NULL, 'S' }, { "output-limit",
				required_argument, NULL, 'O' }, { "input", required_argument,
				NULL, 'i' }, { "delay", required_argument, NULL, 'd' }, {
				"help", no_argument, NULL, 'h' }, { "watcher-data",
				required_argument, NULL, 'w' }, { "var", required_argument,
				NULL, 'v' }, { "solver-data", required_argument, NULL, 'o' }, {
				"timestamp", no_argument, NULL, 1000 }, { "watch-syscalls",
				no_argument, NULL, 1001 },
		{ "use-pty", no_argument, NULL, 1002 }, { "cleanup-own-ipc-queues",
				no_argument, NULL, 1003 }, { "cleanup-all-ipc-queues",
				no_argument, NULL, 1004 }, { "cores", required_argument, NULL,
				1005 }, { "phys-cores", required_argument, NULL, 1006 }, {
				"add-eof", no_argument, NULL, 1007 }, { NULL, no_argument, NULL,
				0 } };

void usage(char *prgname) {
	cout << "Usage: " << prgname << endl
			<< "       [-w file | --watcher-data file]\n"
			<< "       [-v file | --var file]\n"
			<< "       [-o file | --solver-data file]\n"
			<< "       [--cores first-last]\n"
			<< "       [-C cpu-limit | --cpu-limit cpu-limit]\n"
			<< "       [-W time-limit | --wall-clock-limit time-limit]\n"
			<< "       [-M mem-limit | --mem-soft-limit mem-limit]\n"
			<< "       [-S stack-limit | --stack-limit stack-limit]\n"
			<< "       [-d delay | --delay d]\n"
			<< "       [--input filename]\n" << "       [--timestamp]\n"
			<< "       [-O start,max | --output-limit start,max]\n"
			<< "       [--use-pty]\n"
			<< "       [--cleanup-own-ipc-queues | --cleanup-all-ipc-queues]\n"
#ifdef WATCHSYSCALLS
			<< "       [--watch-syscalls]\n"
#endif
			<< "       command\n" << endl;

	cout << "The mem-limit must be expressed in mega-bytes" << endl;
	cout << "The stack-limit must be expressed in mega-bytes" << endl;
	cout << "The cpu-limit must be expressed in seconds (CPU time)" << endl;
	cout << "The time-limit must be expressed in seconds (wall clock time)"
			<< endl;
	cout << "When the time or memory limit is exceeded, the watching "
			<< "process will try to send a SIGTERM and after <delay> "
			<< "seconds will send a SIGKILL to the watched process" << endl;
	cout << "-w filename or --watcher-data filename\n"
			<< "  sends the watcher informations to filename" << endl;
	cout << "-v filename or --var filename\n"
			<< "  save the most relevant information (times,...) in an easy to parse VAR=VALUE file"
			<< endl;
	cout << "-o filename or --solver-data filename\n"
			<< "  redirects the solver output (both stdout and stderr) to filename\n"
			<< "--input filename\n"
			<< "  redirects the standard input of the runned program to filename\n"
			<< "--timestamp\n"
			<< "  instructs to timestamp each line of the solver standard output and\n"
			<< "  error files (which are then redirected to stdout)\n"
			<< "--add-eof\n"
			<< "  when timestamps are used, request to add an 'EOF' line at the end of the solver output\n"
			<< "--output-limit start,max or -O start,max:\n"
			<< "  limits the size of the solver output.\n"
			<< "  Currently implies --timestamp. The solver output will be limited\n"
			<< "  to a maximum of <max> MiB. The first <start> MiB will be\n"
			<< "  preserved as well as the last <max-start> MiB.\n"
			<< "--phys-cores list\n"
			<< "  allocate a subset of the cores to the solver. The list contains\n"
			<< "  core numbers separated by commas, or ranges first-last. This list\n"
			<< "  must contain core identifiers as they are known by the system in\n"
			<< "  /proc/cpuinfo.\n" << "--cores list\n"
			<< "  allocate a subset of the cores to the solver. The list contains\n"
			<< "  core numbers separated by commas, or ranges first-last. This list\n"
			<< "  contains the index of the selected cores in a list of core identifiers\n"
			<< "  sorted by the processor they belong to. For example, if processor 0\n"
			<< "  contains cores 0, 2, 4, 6 and processor 1 contains cores 1, 3, 5, 7,\n"
			<< "  the sorted list is 0, 2, 4, 6, 1, 3, 5, 7 and the argument 0-3 will select\n"
			<< "  the 4 cores of processor 0 (with physical id 0, 2, 4, 6). This option\n"
			<< "  allows to ignore the details of the core numbering scheme used by the kernel.\n"
			<< "--use-pty\n"
			<< "  use a pseudo-terminal to collect the solver output. Currently only\n"
			<< "  available when lines are timestamped. Some I/O libraries (including\n"
			<< "  the C library) automatically flushes the output after each line when\n"
			<< "  the standard output is a terminal. There's no automatic flush when\n"
			<< "  the standard output is a pipe or a plain file. See setlinebuf() for\n"
			<< "  some details. This option instructs runsolver to use a\n"
			<< "  pseudo-terminal instead of a pipe/file to collect the solver\n"
			<< "  output. This fools the solver which will line-buffer its output.\n"
			<< "--cleanup-own-ipc-queues\n"
			<< "  on exit, delete IPC queues that the user owns and to which the solver\n"
			<< "  was the last process to read/write [may fail to delete some queues]\n"
			<< "--cleanup-all-ipc-queues\n"
			<< "  on exit, delete all IPC queues that the user created [will also delete\n"
			<< "  queues that don't belong to the solver]\n" << endl;
#ifdef WATCHSYSCALLS
	cout << "--watch-syscalls intercepts some kernel system calls and checks\n"
	<< "  that the program doesn't make forbidden calls (slows down the\n"
	<< "  execution of the solver)."
	<< endl;
#endif

	exit(1);
}

int main(int argc, char **argv) {
	RunSolver solver;
	int optc;

	// memLimit in KiB
	long memLimit = 0;
	// difference between the 'hard' and the 'soft' limit (in KiB)
	int memSoftToHardLimit = 50 * 1024;

	string cmdline;
	for (int i = 0; i < argc; ++i) {
		cmdline += argv[i];
		cmdline += ' ';
	}

	try {
		ios_base::sync_with_stdio();
		char* set_affinity;
		while ((optc = getopt_long(argc, argv, "+o:w:v:C:W:M:S:O:d:h", longopts,
				NULL)) != EOF) {
			switch (optc) {
			case 'o':
				solver.setOutputRedirection(optarg);
				break;
			case 'i':
				solver.setInputRedirection(optarg);
				break;
			case 'w':
				solver.setWatcherOutputFile(optarg);
				break;
			case 'v':
				solver.setVarOutputFile(optarg);
				break;
			case 'M':
				memLimit = atol(optarg) * 1024;
				break;
			case 'C':
				solver.setCPULimit(atoi(optarg));
				break;
			case 'W':
				solver.setWallClockLimit(atoi(optarg));
				break;
			case 'S':
				solver.addLimit(
						new RunSolver::StackSizeLimit(atoi(optarg) * 1024));
				break;
			case 'd':
				solver.setDelayBeforeKill(atoi(optarg));
				break;
			case 'O':
				int activate, max;
				if (sscanf(optarg, "%d,%d", &activate, &max) != 2
						|| activate >= max) {
					cout << "Syntax: --output-limit A,M with A<M" << endl;
					exit(1);
				}
				solver.setTimeStamping(true);
				solver.setSolverOutputLimits(activate * 1024 * 1024,
						max * 1024 * 1024);
				break;
			case 1000:
				solver.setTimeStamping(true);
				break;
			case 1001:
				solver.watchSyscalls();
				break;
			case 1002:
				solver.setUsePTY(true);
				break;
			case 1003:
				solver.setCleanupSolverOwnIPCQueues(true);
				;
				break;
			case 1004:
				solver.setCleanupAllIPCQueues(true);
				;
				break;
			case 1005:

				set_affinity = getenv("AEATK_SET_TASK_AFFINITY");

				if ((set_affinity != NULL) && strcmp(set_affinity, "1") == 0) {
					printf(
							"[AEATK] Cannot explicitly restrict cores, while also relying on implicit setting\n\n");
					exit(1);
				}

				solver.selectCores(optarg, false);
				break;
			case 1006:
				solver.selectCores(optarg, true);
				break;
			case 1007:
				solver.setTimeStampingAddEOF(true);
				break;
			default:
				usage(argv[0]);
			}
		}

		// this must be output AFTER the command line has been parsed
		// (i.e. after the possible redirection have been set up)
		cout << "runsolver Copyright (C) 2010-2013 Olivier ROUSSEL\n"
		<< "\n"
		<< "This is runsolver version " << version
		<< " (svn: " << SVNVERSION << ") [AEATK2]\n"
		<< "\n"
		<< "This program is distributed in the hope that it will be useful,\n"
		<< "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
		<< "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
		<< "GNU General Public License for more details.\n"
		<< endl;

		if (optind == argc)
			usage(argv[0]);

		cout << "command line: " << cmdline << endl << endl;

		updateCores();

		vector<unsigned short int> cores;

		getAllocatedCoresByProcessorOrder(cores);

		cout << "running on " << cores.size() << " cores: ";
		printAllocatedCores(cout, cores);
		cout << "\n\n";

		if (memLimit)
			solver.setMemLimit(memLimit, memSoftToHardLimit);

		solver.printLimits(cout);

		solver.run(&argv[optind]);
	} catch (exception &e) {
		cout.flush();
		cerr << "\n\tUnexpected exception in runsolver:\n";
		cerr << "\t" << e.what() << endl;
		exit(1);
	}
}

/*
 setitimer(ITIMER_PROF,...) to receive SIGPROF regularly

 alarm to receive SIGALRM at the timeout ???
 */

// Local Variables:
// mode: C++
// End:
