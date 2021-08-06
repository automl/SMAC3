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



#ifndef _SyscallsTracer_hh_
#define _SyscallsTracer_hh_

/* TODO ???
 * check alignment when using PTRACE_PEEKDATA
 *
 * check return value of ptrace
 *
 * investigate the syscall number -1 we get after a sigreturn
 */


#define USE_CLONE_PTRACE

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <map>

#include <unistd.h>
#include <errno.h>

#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/syscall.h>
#include <asm/unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <linux/net.h>

#include "ProcessList.hh"
#include "SyscallNames.hh"

using namespace std;


#ifndef __i386__
#error Sorry, this code is i386 specific
#endif

#ifndef __linux__
#error Sorry, this code is Linux specific
#endif


/**
 * a callback 
 */
class SyscallCallback
{
public:
  virtual void operator() ()=0;
};

/**
 * a class which represent an action to perform when a system call is
 * intercepted
 */
class SyscallAction
{
public:
  typedef
    struct user_regs_struct Registers;

  /**
   * list of system calls that this action should be registered to
   *
   * must return NULL if we don't want to define a list of syscalls to
   * which this action should be registered or otherwise a static array of
   * system calls numbers terminated by -1
   */
  virtual const int *syscallsToRegisterTo()
  {
    return NULL;
  }

  virtual void action(int syscall, bool entering, 
		      pid_t pid, Registers &regs)=0;

  /**
   * helper functions
   */

	// On the i386 architecture, the system call number is put in
	// the register %eax. The arguments to this system call are
	// put into registers %ebx, %ecx, %edx, %esi and %edi, in that
	// order.

  template<typename T> inline T &parm1(Registers &regs)
  {
    return reinterpret_cast<T&>(regs.ebx);
  }

  template<typename T> inline T &parm2(Registers &regs)
  {
    return reinterpret_cast<T&>(regs.ecx);
  }

  template<typename T> inline T &parm3(Registers &regs)
  {
    return reinterpret_cast<T&>(regs.edx);
  }

  template<typename T> inline T &parm4(Registers &regs)
  {
    return reinterpret_cast<T&>(regs.esi);
  }

  template<typename T> inline T &parm5(Registers &regs)
  {
    return reinterpret_cast<T&>(regs.edi);
  }

  template<typename T> inline T returnValue(Registers &regs)
  {
    return static_cast<T>(regs.eax);
  }

  /**
   * import a string from a process
   *
   * maxlen==0 means no limit
   */
  void importString(string &s, pid_t pid, void *addr, int maxlen=0)
  {
    char tab[5]={0,0,0,0,0};
    long &dst=*(long *)tab;

    char *p=static_cast<char *>(addr);

    s.clear();

    do
    {
      dst=ptrace(PTRACE_PEEKDATA,pid,p,NULL);
      s+=tab;
      p+=sizeof(long);
    }
    while(strlen(tab)==4 && (maxlen==0 || s.length()<maxlen));
  }

};


class SyscallActionList 
{
private:
  // list of actions registered
  vector<SyscallAction *> actionList;

  // list of the processes which are currently calling a system call
  // used to determine whether we are entering or exiting the system call
  ProcessList procList;

  struct ExecveInfo
  {
    long from; // from where the call was made
    int n; // number of spurious execve calls

    ExecveInfo()
    {
      from=0;
      n=0;
    }
  };

  // special data for execve
  map<pid_t,ExecveInfo> execveData;

public:
  /**
   * append an action to this action list
   */
  void addAction(SyscallAction *action)
  {
    actionList.push_back(action);
  }

  /**
   * run all actions registered for a given system call
   */
  void run(int syscall, pid_t pid, struct user_regs_struct &regs)
  {
    bool entering=!procList.contains(pid);
    /*
     * ??? if it succeeds execve does not return, we have to find a
     * way to handle this correctly
     */
    if (syscall==SYS_execve)
    {
      // this is a special case !!
      if (entering)
      {
	execveData[pid].from=regs.eip; // remember from where tjis is called
      }
      else
      {
	if (execveData[pid].from!=regs.eip)
	{
	  // expect 2spurious calls, ignore them
	  // ??? is this always the case ?
	  
	  if ((++execveData[pid].n)==2)
	    execveData.erase(pid);

	  return;
	}
      }
	
    }

    if (entering)
      procList.add(pid);

    for(int i=0;i<actionList.size();++i)
      actionList[i]->action(syscall,entering,pid,regs);

    if (!entering)
      procList.remove(pid);
  }
};

/**
 * class which let us intercept system calls
 */
class SyscallsTracer
{
private:
  struct user_regs_struct regs;

  // actions to take when a system call is intercepted
  SyscallActionList *syscallActionLists[nbSyscallNames];

  int nbsyscalls; //???
public:
  SyscallsTracer()
  {
    // no action registered yet
    for(int i=0;i<nbSyscallNames;++i)
    {
      syscallActionLists[i]=NULL;
    }

    nbsyscalls=0; // ???
  }

  ~SyscallsTracer()
  {
    for(int i=0;i<nbSyscallNames;++i)
    {
      if (syscallActionLists[i])
	delete syscallActionLists[i];
    }
  }

  /**
   * register an action for a given system call
   */
  void registerAction(int syscallNumber, SyscallAction *action)
  {
    if (syscallNumber<0 || syscallNumber>nbSyscallNames)
      throw runtime_error("invalid system call number");

    if (syscallActionLists[syscallNumber]==NULL)
    {
      syscallActionLists[syscallNumber]=new SyscallActionList;
    }

    syscallActionLists[syscallNumber]->addAction(action);
  }


  /**
   * register an action to the list of system calls defined by the action
   */
  void registerAction(SyscallAction *action)
  {
    const int *list=action->syscallsToRegisterTo();

    if (list==NULL)
      throw runtime_error("no list of system calls defined inside the action");

    int i;
    for(i=0;list[i]>=0;++i)
      registerAction(list[i],action);

    if(i==0)
      throw runtime_error("list of system calls defined inside the action is empty");
  }

  void childTraceKernelCalls()
  {
    ptrace(PTRACE_TRACEME,0,NULL,NULL);
  }

  void parentTraceKernelCalls(pid_t childpid)
  {
    int status,result;

    result=waitpid(childpid,&status,0);

    if (result!=childpid)
      throw runtime_error("wait for traced child failed");

    if (WIFEXITED(status))
    {
      cout << "\n\nChild to trace (pid=" << childpid << ") exited with status: " 
	   << WEXITSTATUS(status) 
	   << " before we could trace it" << endl;
      exit(2);
    }
    else
      if (WIFSIGNALED(status))
      {
	int sig=WTERMSIG(status);

	cout << "\n\nChild to trace (pid=" << childpid 
	     << ") ended because it received signal " 
	     << sig  << " (" << getSignalName(sig) << ")" 
	     << " before we could trace it" << endl;
	exit(2);
      }

    if (!WIFSTOPPED(status))
      throw runtime_error("traced child is not stopped");

#ifdef debug
    cout << "traced child is first stopped by signal " 
	 << getSignalName(WSTOPSIG(status)) << endl;
#endif

    ptrace(PTRACE_SYSCALL,childpid,NULL,NULL);
  }

  /**
   * trace the system calls of childpid until it exits
   *
   * has almost the same parameters as wait4()
   *
   */
  void watchKernelCalls(pid_t childpid,
			int &childstatus,
			struct rusage &childrusage)
  {
    long err;

    while(true)
    {
      int pid,retry;

      /*
	The wait4() system call below is particularly subject to be
	interrupted by a signal (SIGALRM dor example). The best way to
	avoid it is to use the SA_RESTART flag for sigaction() or the
	siginterrupt() call so that the system call will be
	automatically (and transparently) restarted when interrupted.

	However, we cannot be sure that this will be used so we use a
	loop to relaunch wait4 when it's interrupted.
      */

      retry=0;
      do
      {
	// wait for SIGTRAP from some of our children. Note that we
	// MUST use the __WALL option to get informations from threads
	pid=wait4(-1,&childstatus,__WALL,&childrusage);

#ifdef debug
	if (pid==-1 && errno==EINTR)
	  cout << "WARNING: the internal wait4() system call was interrupted, "
	       << "retrying. Better use sigaction() with "
	       << "SA_NOCLDSTOP|SA_RESTART flags;" << endl;
#endif
      } while(pid==-1 && errno==EINTR 
	      && ++retry<=3); // accept up to 3 interrupts

      if (pid==-1)
      {
	if (errno==ECHILD)
	{
	  // no child any more, our job is finished !
#ifdef tmpdebug
	  cout << "All traced children have exited ! Game is over." << endl;

	  cout << "Intercepted " << nbsyscalls/2 << " system calls" << endl; // ???
#endif
	  return;
	}
	else
	  throw runtime_error(string("wait4 on traced child failed")
			      +strerror(errno));
      }

      if (WIFSTOPPED(childstatus))
      {
	// was it stopped by a system call ?
	if (WSTOPSIG(childstatus)==SIGTRAP)
	{
	  err=ptrace(PTRACE_GETREGS,pid,NULL,&regs);
	  if (err<0)
	  {
	    perror("ptrace(PTRACE_GETREGS,...) failed");
	    continue;
	  }

	  nbsyscalls++; //???
	    
#ifdef debug
	  cout << "[pid=" << pid << "] syscall " 
	       << getSyscallName(regs.orig_eax)
	       << " (orig_eax=" <<regs.orig_eax 
	       << hex << showbase
	       << ",eip=" << regs.eip
	       << noshowbase << dec
	       << ")" << endl;
#endif

	  if (regs.orig_eax<0 || regs.orig_eax>=nbSyscallNames)
	  {
	    if (regs.orig_eax!=-1)
	    {
	      cout << "Ooops ! Got an invalid system call number !"
		   << " (orig_eax=" << regs.orig_eax 
		   << " eax=" << regs.eax << ")" << endl
		   << "Trying to continue..."  << endl; 
	    }
	    /*
	      else ...

	      we get regs.orig_eax==-1 on the "return" of a sigreturn
	      we'll assume for now that we don't have to care about
	      that (after all, strace gets the same value) but this
	      needs to be investigated ???
	     */

	    // restart the process up to the next system call
	    ptrace(PTRACE_SYSCALL,pid,NULL,NULL);

	    continue;
	  }

	  // On the i386 architecture, the system call number is put in
	  // the register %eax. The arguments to this system call are
	  // put into registers %ebx, %ecx, %edx, %esi and %edi, in that
	  // order.

	  if (syscallActionLists[regs.orig_eax])
	    syscallActionLists[regs.orig_eax]->run(regs.orig_eax,pid,regs);

	  // restart the process up to the next system call
	  ptrace(PTRACE_SYSCALL,pid,NULL,NULL);
	}
	else
	{
	  // it wasn't the SIGTRAP, child is not stopped by a system
	  // call. We just have to let the program run until the next
	  // system call
#ifdef debug
	  cout << "Traced child got signal "
	       << getSignalName(WSTOPSIG(childstatus)) 
	       << " just continuing"
	       << endl;
#endif

	  // whatever the signal was, we must restart the process up
	  // to the next system call and transmit that signal
	  ptrace(PTRACE_SYSCALL,pid,NULL,WSTOPSIG(childstatus));
	}
      }
      else
      {
	if (WIFEXITED(childstatus))
	  cout << "One traced child (pid=" << pid << ") exited with status: " 
	       << WEXITSTATUS(childstatus) << endl;
	else
	  if (WIFSIGNALED(childstatus))
	  {
	    int sig=WTERMSIG(childstatus);

	    cout << "One traced child (pid=" << pid 
		 << ") ended because it received signal " 
		 << sig  << " (" << getSignalName(sig) << ")" << endl;
	  }
      }
    }
  }
};


class HeapSizeWatcher : public SyscallAction
{
private:
  // system calls that we should register to
  static const int listOfSyscalls[];

  struct ProcInfo 
  {
    pid_t process; // pid of the process which owns the thread (0 for
		   // a plain process)
    int vsize; // virtual size of this process in Kb (read from /proc/pid/stat)

    int baseBrk; // value of brk to compute the increase of the heap
    int lastBrk; // last value of brk
    long memIncr; // estimated memory increment (bytes)

    ProcInfo()
    {
      process=0;

      vsize=0;

      baseBrk=0;
      lastBrk=0;
      memIncr=0;
    }
  };

  map<pid_t,ProcInfo> data;

  // estimated memory increment (bytes) of all the processes
  long long globalMemIncr; 

  // cumulated size of all processes currentBrkSize (in Kb)
  int cumulatedSize;

  // maximum allowed cumulatedSize (if any). must be zero if inactive
  int maximumCumulatedSize;

  // function to be called when cumulatedSize gets over maximumCumulatedSize
  void (*overQuotaHandlerFn)();

  // functional object to be called when cumulatedSize gets over quota
  SyscallCallback *overQuotaHandlerObj;

  void init()
  {
    cumulatedSize=0;
    maximumCumulatedSize=0;
    globalMemIncr=0;

    overQuotaHandlerFn=NULL;
    overQuotaHandlerObj=NULL;
  }

  /**
   * update the vsize of a process by reading from /proc/pid/stat
   *
   * pid is the id of the process whose vsize must be read again
   * info must be data[pid]
   */
  inline void updateProcessVsize(pid_t pid, ProcInfo &info)
  {
    cumulatedSize-=info.vsize;
    info.vsize=readProcessVSIZE(pid);
    globalMemIncr-=info.memIncr;
    info.memIncr=0;
    cumulatedSize+=info.vsize;
    info.baseBrk=info.lastBrk;
  }

  /**
   * check if we have exceeded our memory quota
   *
   */
  inline void checkOverQuota()
  {
    if (maximumCumulatedSize)
    {
      if (cumulatedSize+(globalMemIncr>>10)<=maximumCumulatedSize)
	return; // we're clearly below our limit

#ifdef debug
      cout << "HeapSizeWatcher::checkOverQuota() is updating processes vsize" 
	   << endl;
#endif

      // update the vsize of all processes with fresh values
      for(map<pid_t,ProcInfo>::iterator it=data.begin();
	  it!=data.end();++it)
	if ((*it).second.memIncr!=0 && (*it).second.process==0)
	  updateProcessVsize((*it).first,(*it).second);

      // now that data is updated, check the limit
      if (cumulatedSize>maximumCumulatedSize)
      {
#ifdef debug
	cout << "memory is over quota" << endl;
#endif

	if (overQuotaHandlerObj)
	  (*overQuotaHandlerObj)();

	if (overQuotaHandlerFn)
	  overQuotaHandlerFn();
      }
    }
  }

public:
  HeapSizeWatcher()
  {
    init();
  }

  HeapSizeWatcher(void (*f)())
  {
    init();
    overQuotaHandlerFn=f;
  }

  HeapSizeWatcher(SyscallCallback *obj)
  {
    init();
    overQuotaHandlerObj=obj;
  }

  /**
   * return the estimated cumulated size of all process (in Kb)
   *
   */
  int getCumulatedSize()
  {
    return cumulatedSize+(globalMemIncr>>10);
  }

  /**
   * max is expressed in Kb
   */
  void setMaximumCumulatedSize(int max)
  {
    maximumCumulatedSize=max;
  }

  void setOverQuotaHandler(void (*f)())
  {
    overQuotaHandlerFn=f;
  }

  void setOverQuotaHandler(SyscallCallback *obj)
  {
    overQuotaHandlerObj=obj;
  }

  virtual const int *syscallsToRegisterTo()
  {
    return listOfSyscalls;
  }

  virtual void action(int syscall, bool entering, pid_t pid, Registers &regs)
  {
#ifdef debug
    cout << "[pid=" << pid << "] ";

    if (entering)
      cout << "entering ";
    else
      cout << "exiting ";

    cout << getSyscallName(syscall) 
	 << " syscall" << endl;
#endif

    pid_t processid=pid; 

    // is this a thread ?
    if (data[processid].process)
      processid=data[processid].process;

    ProcInfo &info=data[processid];

    if (!entering)
    {
      switch(syscall)
      {
      case SYS_brk:
	info.lastBrk=returnValue<int>(regs);

	if (info.baseBrk==0)
	{
	  // we have a process pid and we don't know its initial brk address
#ifdef debug
	  cout << "brk(0)=" 
	       << hex << showbase << returnValue<int>(regs)
	       << noshowbase << dec
	       << endl;
#endif

	  updateProcessVsize(processid,info);
	}
	else
	{
	  info.memIncr+=
	    info.lastBrk-info.baseBrk;
	  globalMemIncr+=
	    info.lastBrk-info.baseBrk;

#ifdef debug
	  cout << "brk()=" 
	       << hex << showbase << returnValue<int>(regs)
	       << noshowbase << dec
	       <<" -> brkSize=" 
	       << ((returnValue<int>(regs)-info.baseBrk)>>10)
	       << endl;
#endif
	}

	checkOverQuota();
	break;

      case SYS_clone:
	if (parm1<long>(regs) & CLONE_VM)
	{
	  // if we have a thread, store the pid of the process which
	  // owns this thread
	  pid_t newpid=returnValue<int>(regs);
	  
	  if (data[pid].process)
	    data[newpid].process=data[pid].process;
	  else
	    data[newpid].process=pid;
	}
	break;

      case SYS_exit:
      case SYS_exit_group:
	if (pid==processid) // if this is a process
	{
	  cumulatedSize-=info.vsize;
	  globalMemIncr-=info.memIncr;
	}
	
	data.erase(pid);
	break;

      case SYS_mmap:
	{
	  long p=parm1<long>(regs);
	  long len;
	  long flags;
	  
	  len=ptrace(PTRACE_PEEKDATA,pid,p+4,NULL);
	  flags=ptrace(PTRACE_PEEKDATA,pid,p+12,NULL);

#ifdef debug
	  cout << "old_mmap(len=" << len 
	       << endl;
#endif
	  info.memIncr+=len;
	  globalMemIncr+=len;
	}
	checkOverQuota();
	break;
	  
      case SYS_mmap2:
	{
	  size_t len=parm2<size_t>(regs);

#ifdef debug
	  cout << "mmap2(len=" << len 
	       << endl;
#endif

	  info.memIncr+=len;
	  globalMemIncr+=len;
	}
	checkOverQuota();
	break;

      case SYS_mremap:
	{
	  size_t oldlen=parm2<size_t>(regs);
	  size_t newlen=parm3<size_t>(regs);
#ifdef debug
	  cout << "mremap(oldlen=" << oldlen 
	       << ",newlen=" << newlen 
	       << endl;
#endif

	  info.memIncr+=newlen-oldlen;
	  globalMemIncr+=newlen-oldlen;

	  if (newlen>oldlen)
	    checkOverQuota();
	}
	break;

      case SYS_munmap:
	{
	  size_t len=parm2<size_t>(regs);
#ifdef debug
	  cout << "munmap(len=" << len 
	       << endl;
#endif

	  // check if munmap is successfull before updating the
	  // counter for otherwise it would be easy to fool us
	  if (returnValue<int>(regs)==0)
	  {
	    info.memIncr+=len;
	    globalMemIncr+=len;
	  }
	}
	break;
      }
    }
#ifdef debug
    cout << "pid=" << processid 
	 << " estimated vsize=" << info.vsize+(info.memIncr>>10)
	 << " /proc/stat vsize=" << readProcessVSIZE(processid)
	 << endl;
#endif
  }

private:
  /**
   * return the process vsize in Kb
   */
  int readProcessVSIZE(int pid)
  {
    char statFileName[64]; // ???
    FILE *file;
    int vsize=0;

    snprintf(statFileName,sizeof(statFileName),"/proc/%d/stat",pid);

    if ((file=fopen(statFileName,"r"))!=NULL)
    {
	  
      fscanf(file,
	     "%*d "
	     "%*s "
	     "%*c "
	     "%*d %*d %*d %*d %*d "
	     "%*lu %*lu %*lu %*lu %*lu "
	     "%*Lu %*Lu %*Lu %*Lu " /* utime stime cu- & cstime */
	     "%*ld %*ld "
	     "%*d "
	     "%*ld "
	     "%*Lu "  /* start_time */
	     "%lu ",&vsize
	     );
      
      fclose(file);

      vsize >>=10;
    }

    return vsize;
  }
};

const int HeapSizeWatcher::listOfSyscalls[]=
  {SYS_brk,
   SYS_mmap,SYS_mmap2,SYS_munmap,SYS_mremap,
   SYS_clone,
   SYS_exit,SYS_exit_group,
   -1};

/**
 * a class to keep a list of processes created by the traced child
 *
 * 
 *
 */
class ProcessWatcher : public SyscallAction
{
private:
  // system calls that we should register to
  static const int listOfSyscalls[];

  // list of processes that we must maintain (to be provided by the
  // user of this class)
  ProcessList &procList;

public:
  ProcessWatcher(ProcessList &list) : procList(list) {}

  virtual const int *syscallsToRegisterTo()
  {
    return listOfSyscalls;
  }

  virtual void action(int syscall, bool entering, pid_t pid, Registers &regs)
  {
#ifdef debug
    cout << "[pid=" << pid << "]  ";

    if (entering)
      cout << "entering ";
    else
      cout << "exiting ";

    cout << getSyscallName(syscall) << " syscall" << endl;
#endif

    if (!entering && syscall==SYS_clone)
    {
      // trace the new child
      pid_t newpid=returnValue<int>(regs);

      if (parm1<long>(regs) & CLONE_VM)
      {
	cout << "New thread pid=" << newpid << endl;
      }
      else
      {
        // add new pid to process list -- if only it is a process
	// and not a thread
        procList.add(newpid);

	cout << "New process pid=" << newpid << endl;
      }

      if (newpid>0)
      {
#ifndef USE_CLONE_PTRACE
	ptrace(PTRACE_ATTACH,newpid,NULL,NULL);

	int status,result;

	result=waitpid(newpid,&status,__WALL);

	if (result!=newpid)
	{
	  throw runtime_error(string("wait for traced child after a clone() failed (")+strerror(errno)+")");
	}

	if (!WIFSTOPPED(status))
	  throw runtime_error("trace child after a clone() is not stopped");

	cout << "traced child after a clone() is first stopped by signal " 
	     << getSignalName(WSTOPSIG(status)) << endl;

	ptrace(PTRACE_SYSCALL,newpid,NULL,NULL);
#endif
      }
    }     
    else
      if (entering)
      {
	switch(syscall)
	{
	case SYS_exit:
	case SYS_exit_group:
	  {
	    // remove pid of the caller from process list
	    procList.remove(pid);

#ifdef debug
	    cout << "unregistering process " << pid << endl;
#endif
	  }
	  break;

	case SYS_execve:
	  {
	    string exec;

	    importString(exec,pid,parm1<char *>(regs),200);
	    // do we need that limit ???

	    cout << "execve syscall for " 
		 << exec
		 << " executable"
		 << endl;
	  }
	  break;

#ifdef USE_CLONE_PTRACE
	case SYS_clone:
	  {
#ifdef debug
	    cout << "forcing CLONE_PTRACE flag" << endl;
#endif
	    // modify the clone parameters to add the CLONE_PTRACE flag
	    parm1<int>(regs)|=CLONE_PTRACE;
	    ptrace(PTRACE_SETREGS,pid,NULL,&regs);
	  }
	  break;
#endif
	}
      }
  }

};

const int ProcessWatcher::listOfSyscalls[]=
  {SYS_clone,
   SYS_exit,SYS_exit_group,
   SYS_execve,
   -1};

/**
 * a class to watch file accesses
 */
class FileWatcher : public SyscallAction
{
private:
  // system calls that we should register to
  static const int listOfSyscalls[];

public:
  FileWatcher()
  {
  }

  virtual const int *syscallsToRegisterTo()
  {
    return listOfSyscalls;
  }

  virtual void action(int syscall, bool entering, pid_t pid, Registers &regs)
  {
#ifdef debug
    cout << "[pid=" << pid << "] ";

    if (entering)
      cout << "entering ";
    else
      cout << "exiting ";

    cout << getSyscallName(syscall) << " syscall: parm1=" 
	 << hex << showbase << parm1<int>(regs) 
	 << dec << endl;
#endif

    if (entering)
    {
      string filename;
      importString(filename,pid,(void *)parm1<int>(regs));

      cout << "open syscall for file " << filename << endl;
    }
    else
    {
      //if (true)
      //  changeResultValue(regs,EACCES);
    }
  }
};

const int FileWatcher::listOfSyscalls[]={SYS_open,-1};


/**
 * a class to watch network activity
 */
class NetworkWatcher : public SyscallAction
{
private:
  // system calls that we should register to
  static const int listOfSyscalls[];

  // number of parameters for each subcall
  static const int nbSubcallParms[];
  static const int maxNbParms;

public:
  NetworkWatcher()
  {
  }

  virtual const int *syscallsToRegisterTo()
  {
    return listOfSyscalls;
  }

  virtual void action(int syscall, bool entering, pid_t pid, Registers &regs)
  {
#ifdef debug
    cout << "[pid=" << pid << "] ";

    if (entering)
      cout << "entering ";
    else
      cout << "exiting ";

    cout << getSyscallName(syscall) << " syscall: parm1=" 
	 << hex << showbase << parm1<int>(regs) 
	 << dec << endl;
#endif

    // parameters of the subcall
    long parms[maxNbParms];

    int subcall=parm1<int>(regs);

    if (entering)
    {
      cout << "socket syscall " << flush;

      getparms(parms,pid,subcall,regs);

      switch(subcall)
      {
      case SYS_SOCKET:
	decodeSocket(parms);
	break;
      case SYS_BIND:
	decodeSockSockaddrSocklen("bind",pid,parms);
	cout << endl;
	break;
      case SYS_CONNECT:
	decodeSockSockaddrSocklen("connect",pid,parms);
	cout << endl;
	break;
      case SYS_LISTEN:
	cout << "listen(" << parms[0] << "," << parms[1] << ")" << endl;
	break;
      case SYS_ACCEPT:
	cout << "accept(" << parms[0] << ",...)" << endl;
	break;
      case SYS_GETSOCKNAME:
	cout << "getsockname(...)" << endl;
	break;
      case SYS_GETPEERNAME:
	cout << "getpeername(...)" << endl;
	break;
      case SYS_SOCKETPAIR:
	cout << "socketpair(...)" << endl;
	break;
      case SYS_SEND:
	cout << "send(...)" << endl;
	break;
      case SYS_RECV:
	cout << "recv(...)" << endl;
	break;
      case SYS_SENDTO:
	cout << "sendto(" << parms[0] << ",buf=...,len=" << parms[2]
	     << ",flags=" << parms[3] << ",";
	if (parms[4]!=0)
	  decodeSockAddr(pid,parms[4],parms[5]);
	else
	  cout << "NULL";
	cout << ",len=" << parms[5] << ")" << endl;
	break;
      case SYS_RECVFROM:
	cout << "recvfrom(...)" << endl;
	break;
      case SYS_SHUTDOWN:
	cout << "shutdown(...)" << endl;
	break;
      case SYS_SETSOCKOPT:
	cout << "setsockopt(...)" << endl;
	break;
      case SYS_GETSOCKOPT:
	cout << "getsockopt(...)" << endl;
	break;
      case SYS_SENDMSG:
	cout << "sendmsg(...)" << endl;
	break;
      case SYS_RECVMSG:
	cout << "recvmsg(...)" << endl;
	break;
      default:
	cout << "unknown syscall(...)" << endl;
	break;
      }
    }
    else
    {
      switch(subcall)
      {
      case SYS_SOCKET:
	cout << "=" << returnValue<int>(regs) << endl;
	break;
      }
    }
  }

private:
  void getparms(long *parms, pid_t pid, int subcall, Registers &regs)
  {
    long p=parm2<long>(regs);
    for(int i=0;i<nbSubcallParms[subcall];++i,p+=sizeof(long))
    {
      parms[i]=ptrace(PTRACE_PEEKDATA,pid,p,NULL);
      if (parms[i]==-1 && errno!=0)
	throw runtime_error("getparms failed");
    }
  }


  /**
   * decode a sockaddr structure in the traced process memory space
   */
  void decodeSockAddr(pid_t pid, long sockaddr, long len)
  {
    if (len<1)
      return;

    long sock[len/sizeof(long)+1];

    for(int i=0;i<=len/sizeof(long);++i)
      sock[i]=ptrace(PTRACE_PEEKDATA,pid,sockaddr+i*sizeof(long),NULL);

    cout << "{sa_family=";
    switch(((struct sockaddr *)sock)->sa_family)
    {
    case AF_UNIX:
      cout << "AF_UNIX,path="
	   << ((struct sockaddr_un *)sock)->sun_path;
      break;
    case AF_INET:
      cout << "AF_INET,sin_addr="
	   << inet_ntoa(((struct sockaddr_in *)sock)->sin_addr)
	   << ",sin_port=" << ntohs(((struct sockaddr_in *)sock)->sin_port);
      break;
    case AF_INET6:
      cout << "AF_INET6";
      break;
    default:
      cout << "???";
      break;
    }

    cout << "}";
  }

  void decodeSocket(long *parms)
  {
    cout << "socket(";
    switch(parms[0])
    {
    case PF_UNIX:
      cout << "PF_UNIX";
      break;
    case PF_INET:
      cout << "PF_INET";
      break;
    case PF_INET6:
      cout << "PF_INET6";
      break;
    case PF_IPX:   
      cout << "PF_IPX";
      break;
    case PF_NETLINK:
      cout << "PF_NETLINK";
      break;
    case PF_X25:
      cout << "PF_X25";
      break;
    case PF_AX25:  
      cout << "PF_AX25";
      break;
    case PF_ATMPVC: 
      cout << "PF_ATMPVC";
      break;
    case PF_APPLETALK:
      cout << "PF_APPLETALK";
      break;
    case PF_PACKET:
      cout << "PF_PACKET";
      break;
    default:
      cout << "??? (" << hex 
	   << showbase << parms[0] 
	   << noshowbase << dec << ")";
      break;
    }
    cout << ",";
    switch(parms[1])
    {
    case SOCK_STREAM:
      cout << "SOCK_STREAM";
      break;
    case SOCK_DGRAM:
      cout << "SOCK_DGRAM";
      break;
    case SOCK_SEQPACKET:
      cout << "SOCK_SEQPACKET";
      break;
    case SOCK_RAW:   
      cout << "SOCK_RAW";
      break;
    case SOCK_RDM:
      cout << "SOCK_RDM";
      break;
    case SOCK_PACKET:
      cout << "SOCK_PACKET";
      break;
    default:
      cout << "??? (" << hex 
	   << showbase << parms[0] 
	   << noshowbase << dec << ")";
      break;
    }
    cout << "," << parms[2] << ")";
  }

  void decodeSockSockaddrSocklen(string fn,pid_t pid, long *parms)
  {
    cout << fn << "(";
    cout << parms[0] << ",";
    decodeSockAddr(pid,parms[1],parms[2]);
    cout << "," << parms[2] << ")";
  }

};

const int NetworkWatcher::listOfSyscalls[]={SYS_socketcall,-1};

// see also NetworkWatcher::maxNbParms
const int NetworkWatcher::nbSubcallParms[]=
  { 2, // socketcall
    3, // socket
    3, // bind  
    3, // connect
    2, // listen     
    3, // accept     
    3, // getsockname
    3, // getpeername
    4, // socketpair 
    4, // send
    4, // recv
    6, // sendto
    6, // recvfrom
    2, // shutdown
    5, // setsockopt
    5, // getsockopt
    5, // sendmsg
    5  // recvmsg
  };
 
// greatest number in the array above
const int NetworkWatcher::maxNbParms=6;

//asmlinkage long sys_socketcall(int call, unsigned long __user *args)
//{
//	unsigned long a[6];
//	unsigned long a0,a1;
//	int err;
//
//	if(call<1||call>SYS_RECVMSG)
//		return -EINVAL;
//
//	/* copy_from_user should be SMP safe. */
//	if (copy_from_user(a, args, nargs[call]))
//		return -EFAULT;
//		
//	a0=a[0];
//	a1=a[1];
//	
//	switch(call) 
//	{
//		case SYS_SOCKET:
//			err = sys_socket(a0,a1,a[2]);
//			break;
//		case SYS_BIND:
//			err = sys_bind(a0,(struct sockaddr __user *)a1, a[2]);
//			break;
//		case SYS_CONNECT:
//			err = sys_connect(a0, (struct sockaddr __user *)a1, a[2]);
//			break;
//		case SYS_LISTEN:
//			err = sys_listen(a0,a1);
//			break;
//		case SYS_ACCEPT:
//			err = sys_accept(a0,(struct sockaddr __user *)a1, (int __user *)a[2]);
//			break;
//		case SYS_GETSOCKNAME:
//			err = sys_getsockname(a0,(struct sockaddr __user *)a1, (int __user *)a[2]);
//			break;
//		case SYS_GETPEERNAME:
//			err = sys_getpeername(a0, (struct sockaddr __user *)a1, (int __user *)a[2]);
//			break;
//		case SYS_SOCKETPAIR:
//			err = sys_socketpair(a0,a1, a[2], (int __user *)a[3]);
//			break;
//		case SYS_SEND:
//			err = sys_send(a0, (void __user *)a1, a[2], a[3]);
//			break;
//		case SYS_SENDTO:
//			err = sys_sendto(a0,(void __user *)a1, a[2], a[3],
//					 (struct sockaddr __user *)a[4], a[5]);
//			break;
//		case SYS_RECV:
//			err = sys_recv(a0, (void __user *)a1, a[2], a[3]);
//			break;
//		case SYS_RECVFROM:
//			err = sys_recvfrom(a0, (void __user *)a1, a[2], a[3],
//					   (struct sockaddr __user *)a[4], (int __user *)a[5]);
//			break;
//		case SYS_SHUTDOWN:
//			err = sys_shutdown(a0,a1);
//			break;
//		case SYS_SETSOCKOPT:
//			err = sys_setsockopt(a0, a1, a[2], (char __user *)a[3], a[4]);
//			break;
//		case SYS_GETSOCKOPT:
//			err = sys_getsockopt(a0, a1, a[2], (char __user *)a[3], (int __user *)a[4]);
//			break;
//		case SYS_SENDMSG:
//			err = sys_sendmsg(a0, (struct msghdr __user *) a1, a[2]);
//			break;
//		case SYS_RECVMSG:
//			err = sys_recvmsg(a0, (struct msghdr __user *) a1, a[2]);
//			break;
//		default:
//			err = -EINVAL;
//			break;
//	}
//	return err;
//}

#endif

// Local Variables:
// mode: C++
// End:

