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



#ifndef _ProcessTree_hh_
#define _ProcessTree_hh_

#include <sys/types.h>
#include <dirent.h>
#include <signal.h>
#include <errno.h>

#include <iostream>
#include <cctype>
#include <map>
#include <stdexcept>

#include "ProcessData.hh"

using namespace std;

class ProcessTree
{
private:
  typedef map<pid_t,ProcessData *> ProcMap;
  
  ProcMap tree;
  pid_t currentRootPID;

  char loadavgLine[1024]; // ???
  long memTotal,memFree,swapTotal,swapFree; // data from /proc/meminfo

  float elapsed; // number of seconds elapsed since the start of the program

  float uptime; // up time of the host (in seconds)

  bool treeHasAllProcesses;

  // process groups of the solver
  set<pid_t> solverProcessGroups;

  // process group id of runsolver itself
  pid_t runsolverGroupId;

public:
  ProcessTree()
  {
    currentRootPID=0;
    treeHasAllProcesses=false;

    runsolverGroupId=getpgrp();
  }

  ProcessTree(const ProcessTree &pt)
  {
    currentRootPID=pt.currentRootPID;
    treeHasAllProcesses=false;

    elapsed=pt.elapsed;
    memTotal=pt.memTotal;
    memFree=pt.memFree;
    swapTotal=pt.swapTotal;
    swapFree=pt.swapFree;

    clone(pt,currentRootPID);
    strncpy(loadavgLine,pt.loadavgLine,sizeof(pt.loadavgLine));
  }

  ~ProcessTree()
  {
    clear();
  }

  void clear()
  {
    for(ProcMap::iterator it=tree.begin();it!=tree.end();++it)
      delete (*it).second;

    tree.clear();
  }

  void setDefaultRootPID(pid_t pid)
  {
    currentRootPID=pid;
  }


  void setElapsedTime(float timeSinceStartOfProgram)
  {
    elapsed=timeSinceStartOfProgram;
  }

  float getElapsedTime() const
  {
    return elapsed;
  }

  /**
   * gather informations on all processes (to determine all children
   * of the watched process)
   */
  void readProcesses()
  {
    readProcesses(currentRootPID);
  }

  /**
   * update informations on processes which are a child of the current
   * root.
   *
   * doesn't attempt to identify new children processes
   */
  void updateProcessesData()
  {
    updateProcessesData(currentRootPID);
  }

  /**
   * return true when the main process has ended
   */
  bool rootProcessEnded()
  {
    ProcMap::iterator it=tree.find(currentRootPID);

    if (it!=tree.end() && (*it).second!=NULL)
      return false; // no, still running

#if 0
    // the main process terminated, but some children may still be
    // running. Look for a new root process (with init as parent and
    // the process group id given to the solver)

    // get a fresh list of processes
    readProcesses();

    for(ProcMap::iterator it=tree.begin();it!=tree.end();++it)
    {
      ProcessData *data=(*it).second;

      if (!data)
      {
	//cout << "Ooops ! No data available on process " << (*it).first << endl;
	continue;
      }

      if (data->getppid()==1 && data->getProcessGroupId()==currentRootPID)
      {
	// we have found the new root
	currentRootPID=(*it).first;
	cout << "New process root " << currentRootPID << endl;
	return false;
      }
    }
#endif

    return true;
  }

  /**
   * gather informations on all processes (to determine all children
   * of the watched process)
   *
   */
  void readProcesses(pid_t root)
  {
    DIR *procfs=opendir("/proc");
    struct dirent *dirEntry;
    pid_t pid;

    clear();
    solverProcessGroups.clear();

    currentRootPID=root;

    readGlobalData();

    if (!procfs)
      throw runtime_error("unable to read /proc filesystem");

    while((dirEntry=readdir(procfs)))
    {
      // we only care about process ID
      if (!isdigit(*dirEntry->d_name))
	continue;

      //cout << "process " << dirEntry->d_name << endl;

      pid=atoi(dirEntry->d_name);
      tree[pid]=new ProcessData(pid);
    }

    closedir(procfs);

    treeHasAllProcesses=true;

    identifyChildren();

    readTasksRec(root);
  }
  
  /**
   * update informations on processes which are a child of root.
   *
   * doesn't attempt to identify new children processes
   */
  void updateProcessesData(pid_t root)
  {
    ProcessData *data=tree[root];

    if (!data) // no data on this process
      return;

    if (!data->update())
    {
      // this process doesn't exist any more
      tree.erase(root);
      return;
    }

    for(int i=0;i<data->getNbChildren();++i)
    {
      pid_t childpid=data->getPIDChild(i);
      updateProcessesData(childpid);
    }

  }
  /**
   *
   * list heavy processes running on the system. must be called right
   * after readProcesses. We only consider processes which are run
   * by another user.
   *
   * threshold is the percentage of CPU (between 0 and 1) above which
   * a process is considered as heavy
   */
  void dumpHeavyProcesses(ostream &s, float threshold)
  {
    uid_t myUid=getuid();

    // we need all processes in the tree. Reread if necessary.
    if (!treeHasAllProcesses)
      readProcesses();

    cout << "heavy processes:\n";

    for(ProcMap::iterator it=tree.begin();it!=tree.end();++it)
    {
      ProcessData *data=(*it).second;

      if (!data)
      {
	cout << "Ooops ! No data available on process " << (*it).first << endl;
	continue;
      }

      float pcpu=data->percentageCPU(uptime);

      // is this someone else process which uses a significant
      // proportion of the CPU?
      if (data->getUid()!=myUid && pcpu>threshold)
      {
	pid_t pid=(*it).first;

	s << "  %CPU=" << static_cast<int>(pcpu*100) 
	  << " pid=" << pid 
	  << " uid=" << data->getUid() 
	  << " cmd=";

	dumpCmdLine(s,pid);

	s << endl;
      }
    }
  }

  float currentCPUTime()
  {
    float userTime=0,systemTime=0;
    currentCPUTimeRec(currentRootPID,userTime,systemTime);

    return userTime+systemTime;
  }

  void currentCPUTime(float &userTime, float &systemTime)
  {
    userTime=0;
    systemTime=0;
    currentCPUTimeRec(currentRootPID,userTime,systemTime);
  }

  long currentVSize()
  {
    return currentVSizeRec(currentRootPID);
  }

  /**
   * add the pid of each solver task to "list"
   */
  void listProcesses(set<pid_t> &list)
  {
    listProcessesRec(list,currentRootPID);
  }

  void dumpProcessTree(ostream &out)
  {
    cout << "\n[startup+" << elapsed << " s]" << endl;

    dumpGlobalData(out);

    dumpProcessTreeRec(out,currentRootPID);
  }

  void dumpCPUTimeAndVSize(ostream &out)
  {
    float userTime,systemTime,VSize;
    VSize=currentVSize();
    currentCPUTime(userTime,systemTime);
    dumpCPUTimeAndVSize(out,userTime+systemTime,VSize);
  }

  void dumpCPUTimeAndVSize(ostream &out, 
			   float currentCPUTime, float currentVSize)
  {
    cout << "Current children cumulated CPU time (s) " 
	 << currentCPUTime << endl;

    cout << "Current children cumulated vsize (KiB) " 
	 << static_cast<long>(currentVSize+0.5) << endl;
  }

  /**
   * send a signal to the whole process tree without delay
   */
  void sendSignalNow(int sig)
  {
    if(currentRootPID)
      sendSignalNowRec(currentRootPID,sig);
  }

  /**
   * send a signal to the whole process tree without delay
   */
  void sendSignalToProcessGroups(int sig)
  {
    for(set<pid_t>::iterator it=solverProcessGroups.begin();
	it!=solverProcessGroups.end();++it)
      kill(-(*it),sig);
  }

  void sendSignalBottomUp(int sig)
  {
    if(currentRootPID)
      sendSignalBottomUpRec(currentRootPID,sig);
  }

  void sendSignalBottomUp(pid_t pid, int sig)
  {
    sendSignalBottomUpRec(pid,sig);
  }

protected:
  void readGlobalData()
  {
    FILE *file;

    if ((file=fopen("/proc/loadavg","r"))!=NULL)
    {
      fgets(loadavgLine,sizeof(loadavgLine),file);
      fclose(file);
    }

    if ((file=fopen("/proc/meminfo","r"))!=NULL)
    {
      fscanf(file,
#if WSIZE==32
	     "%*s%d%*s"  // MemTotal:      1033624 kB
	     "%*s%d%*s"  // MemFree:         13196 kB
	     "%*s%*d%*s"   // Buffers:          8084 kB
	     "%*s%*d%*s"   // Cached:         343436 kB
	     "%*s%*d%*s"   // SwapCached:          0 kB
	     "%*s%*d%*s"   // Active:         803400 kB
	     "%*s%*d%*s"   // Inactive:       154436 kB
	     "%*s%*d%*s"   // HighTotal:      130240 kB
	     "%*s%*d%*s"   // HighFree:          120 kB
	     "%*s%*d%*s"   // LowTotal:       903384 kB
	     "%*s%*d%*s"   // LowFree:         13076 kB
	     "%*s%d%*s"  // SwapTotal:     2048248 kB
	     "%*s%d%*s"  // SwapFree:      2041076 kB
#else
	     "%*s%ld%*s"  // MemTotal:      1033624 kB
	     "%*s%ld%*s"  // MemFree:         13196 kB
	     "%*s%*d%*s"   // Buffers:          8084 kB
	     "%*s%*d%*s"   // Cached:         343436 kB
	     "%*s%*d%*s"   // SwapCached:          0 kB
	     "%*s%*d%*s"   // Active:         803400 kB
	     "%*s%*d%*s"   // Inactive:       154436 kB
	     "%*s%*d%*s"   // HighTotal:      130240 kB
	     "%*s%*d%*s"   // HighFree:          120 kB
	     "%*s%*d%*s"   // LowTotal:       903384 kB
	     "%*s%*d%*s"   // LowFree:         13076 kB
	     "%*s%ld%*s"  // SwapTotal:     2048248 kB
	     "%*s%ld%*s"  // SwapFree:      2041076 kB
#endif
	     ,&memTotal,&memFree,&swapTotal,&swapFree);

      fclose(file);
    }

    if ((file=fopen("/proc/uptime","r"))!=NULL)
    {
      fscanf(file,
	     "%g"  // uptime
	     ,&uptime);

      fclose(file);
    }
  }

  void identifyChildren()
  {
    // get links from fathers to children
    for(ProcMap::iterator it=tree.begin();it!=tree.end();++it)
    {
      ProcessData *data=(*it).second;

      if (!data)
      {
	cout << "Ooops ! No data available on process " << (*it).first << endl;
	continue;
      }

      pid_t parent=data->getppid();

      if (parent==-1)
	continue; // we just have no data on this process

      ProcMap::iterator itParent=tree.find(parent);
      if (itParent!=tree.end())
	(*itParent).second->addChild((*it).first);
#ifdef debug
      else
	if ((*it).first!=1) // init has no father
	{
	  cout << "Ooops! Can't find parent pid " << parent 
	       << " of child pid " <<  (*it).first << endl;
	  dumpProcessTree(cout);
	}
#endif
    }
  }


  void dumpGlobalData(ostream &out)
  {
    out << "/proc/loadavg: " << loadavgLine;
    out << "/proc/meminfo: memFree=" << memFree << "/" << memTotal
	<< " swapFree=" << swapFree << "/" << swapTotal << endl;
  }

  void currentCPUTimeRec(pid_t pid, float &userTime, float &systemTime)
  {
    ProcessData *data=tree[pid];

    if (!data) // no data on this process
      return;

    userTime+=data->getOverallUserTime();
    systemTime+=data->getOverallSystemTime();

    for(int i=0;i<data->getNbChildren();++i)
    {
      pid_t childpid=data->getPIDChild(i);
      if (tree[childpid] && !tree[childpid]->isTask())
	currentCPUTimeRec(childpid,userTime,systemTime);
    }
  }

  long currentVSizeRec(pid_t pid)
  {
    ProcessData *data=tree[pid];

    if (!data) // no data on this process
      return 0;

    long size=data->getVSize();

    for(int i=0;i<data->getNbChildren();++i)
    {
      pid_t childpid=data->getPIDChild(i);
      if (tree[childpid] && !tree[childpid]->isTask())
	size+=currentVSizeRec(childpid);
    }

    return size;
  }

  void sendSignalNowRec(pid_t pid, int sig)
  {
    ProcessData *data=tree[pid];

    if (!data) // no data on this process
      return;

    if (data->getNbChildren()!=0)
    {
      for(int i=0;i<data->getNbChildren();++i)
      {
	pid_t childpid=data->getPIDChild(i);
	if (tree[childpid] && !tree[childpid]->isTask())
	  sendSignalNowRec(childpid,sig);
      }
    }

    kill(pid,sig);
  }

  void sendSignalBottomUpRec(pid_t pid, int sig)
  {
    ProcessData *data=tree[pid];

    if (!data) // no data on this process
      return;

    if (data->getNbChildren()!=0)
    {
      for(int i=0;i<data->getNbChildren();++i)
      {
	pid_t childpid=data->getPIDChild(i);
	if (tree[childpid] && !tree[childpid]->isTask())
	  sendSignalBottomUpRec(childpid,sig);
      }

      // give some time to the father to wait for its children
      struct timespec delay={0,020000000}; // 20 ms
      
      // use a loop in case of an interrupt
      while(nanosleep(&delay,&delay)==-1 && errno==EINTR);
    }

    kill(pid,sig);
  }

  void readTasksRec(pid_t pid)
  {
    ProcessData *data=tree[pid];

    if (!data) // no data on this process
      return;

    pid_t groupId=data->getProcessGroupId();

    if(groupId!=runsolverGroupId)
      solverProcessGroups.insert(groupId);

    readProcessTasks(pid);

    int nb=data->getNbChildren();
    for(int i=0;i<nb;++i)
    {
      int childpid=data->getPIDChild(i);
      if (tree[childpid] && !tree[childpid]->isTask())
	readTasksRec(childpid);
    }
  }

  void readProcessTasks(pid_t pid)
  {
    char processdir[64]; // ???
    DIR *procfs;
    struct dirent *dirEntry;
    pid_t tid;
    ProcessData *data;

    data=tree[pid];

    if (!data)
      return;

    snprintf(processdir,sizeof(processdir),"/proc/%d/task",pid);

    procfs=opendir(processdir);
    if (!procfs)
    {
      if (errno==ENOENT)
      {
	// process "pid" is probably gone. Don't make a fuss about it
	return;
      }

      cout << "!!! unable to read " << processdir << " filesystem (" 
	   << strerror(errno) << ") !!!" << endl;
      return;
    }

    while((dirEntry=readdir(procfs)))
    {
      // we only care about process ID
      if (!isdigit(*dirEntry->d_name))
	continue;

      tid=atoi(dirEntry->d_name);
      if (tid==pid)
	continue;

      //cout << "task " << dirEntry->d_name 
      //     << " (pid=" << pid << ")" << endl;

      tree[tid]=new ProcessData(pid,tid);

      // add a link from the father to the task
      data->addChild(tid);
    }

    closedir(procfs);
  }

  void listProcessesRec(set<pid_t> &list,pid_t pid)
  {
    ProcessData *data=tree[pid];
    
    if (!data)
      return;

    list.insert(pid);

    for(int i=0;i<data->getNbChildren();++i)
      listProcessesRec(list,data->getPIDChild(i));
  }

  void dumpProcessTreeRec(ostream &out,pid_t pid)
  {
    ProcessData *data=tree[pid];
    
    if (!data)
      return;

    out << *data;
    for(int i=0;i<data->getNbChildren();++i)
      dumpProcessTreeRec(out,data->getPIDChild(i));
  }

  void clone(const ProcessTree &pt, pid_t pid)
  {
    treeHasAllProcesses=false; // we only copy the solver processes

    ProcMap::const_iterator it=pt.tree.find(pid);
    if (it==pt.tree.end())
      return;

    ProcessData *data=(*it).second;
    
    if (!data)
      return;

    tree[pid]=new ProcessData(*data);

    for(int i=0;i<data->getNbChildren();++i)
      clone(pt,data->getPIDChild(i));
  }

  void dumpCmdLine(ostream &s, pid_t pid)
  {
    char buffer[128];
    char fileName[64]; // ???
    int fd;

    snprintf(fileName,sizeof(fileName),"/proc/%d/cmdline",pid);
    
    fd=open(fileName,O_RDONLY);

    if(fd>0)
    {
      unsigned int size=0,r;

      while(size<sizeof(buffer) && 
	    (r=read(fd,buffer+size,sizeof(buffer)-size))>0)
	size+=r;

      for(unsigned int i=0;i<size;++i)
	if(buffer[i])
	  s << buffer[i];
	else
	  s << ' ';

      close(fd);
    }
  }

};

// Local Variables:
// mode: C++
// End:
#endif
