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



#ifndef _TimeStamper_hh_
#define _TimeStamper_hh_

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>

// if we use the TimeStamper in a threaded program, we may have to use
// mutexes
#include <pthread.h> 

#include "CircularBufferFilter.hh"

using namespace std;

/**
 * a class that intercepts the data on some watched file descriptors
 * and adds a timestamp at the beginning of each line.
 */
class TimeStamper
{
public:
  TimeStamper(bool withCPUtime=true) : withCPUtime(withCPUtime)
  {
    // initialize select() data
    FD_ZERO(&watcher);
    max=1;

    // in case the user forgets to call it
    resetTimeStamp();

    lastKnownCPUtime=0;
    pthread_mutex_init(&cputimeMutex,NULL);

    addEOFLine=false;

    incompleteLineSent=false;
 }

  /**
   * specifies if a special 'EOF' line should be added at the end of
   * output. This line allows to get the value of the timestamps when
   * the process ends.
   *
   * @parm add: true iff the 'EOF' line must be added
   */
  void addEOF(bool add)
  {
    addEOFLine=add;
  }

  /**
   * add a file descriptor (inputfd) to the list of timestamped files
   *
   * @parm inputfd: file descriptor of the file that must be timestamped
   * @parm letter: one letter name of the timestamped stream (0 if unused)
   * @parm outputfd: file descriptor to use to output the
   *                 timestamped stream (stdout by default)
   */
  void watch(int inputfd, char letter=0, int outputfd=STDOUT_FILENO)
  {
    FD_SET(inputfd,&watcher);
    if(inputfd>=max)
      max=inputfd+1;

    watched.push_back(info(inputfd,outputfd,letter,true));
  }

  /**
   * add a file descriptor (inputfd) to the list of timestamped files
   *
   * @parm inputfd: file descriptor of the file that must be timestamped
   * @parm letter: one letter name of the timestamped stream (0 if unused)
   * @parm outputfilter: filter to use to output the timestamped stream 
   */
  void watch(int inputfd, AbstractFilter *outputfilter, char letter=0)
  {
    FD_SET(inputfd,&watcher);
    if(inputfd>=max)
      max=inputfd+1;

    watched.push_back(info(inputfd,outputfilter,letter,true));
  }

  /** 
   * reset the time stamp
   */
  void resetTimeStamp()
  {
    gettimeofday(&tvStart,NULL);
  }

  /**
   * obtain the current time stamp
   *
   * @return the current timestamp (for external use)
   */
  struct timeval getTimeStamp()
  {
    gettimeofday(&tv,NULL);
    tv.tv_usec-=tvStart.tv_usec;
    tv.tv_sec-=tvStart.tv_sec;
    if(tv.tv_usec<0)
    {
      tv.tv_sec-=1;
      tv.tv_usec+=1000000;
    }

    return tv;
  }

  /**
   * loop that waits for data on one of the watched files
   * and timestamps each line read from these files
   *
   * the loop ends when all watched files are closed
   *
   * should be called from a separate thread of process (in most cases)
   */
  void timeStampLines()
  {
    fd_set tmp;
    int result,watchDog=0;

    while (watched.size())
    {
      tmp=watcher;

      // wait for data to become available on one of the watched files
      result=select(max,&tmp,NULL,NULL,NULL);

      if(result<0)
      {
        if(++watchDog<10)
          continue;
        else
        {
          cout << "Error in TimeStamper::timeStampLines(), select() keeps returning errors, exiting." << endl;
          break;
        }
      }

      watchDog=0;

      for(size_t i=0;i<watched.size();)
      {
	int fd=watched[i].inputdescr;
	bool del=false;

	if(FD_ISSET(fd,&tmp))
	  if(!readFrom(i)) // read data and timestamp lines
	    {
	      // EOF: remember to remove the file from our watch list
	      del=true;
	    }

	if(del)
	{
	  // remove the file from the select() list
	  FD_CLR(fd,&watcher);

	  // remove the file from our list
	  watched.erase(watched.begin()+i);

	  // recompute max
	  max=1;
	  for(size_t i=0;i<watched.size();++i)
	    if(watched[i].inputdescr>=max)
	      max=watched[i].inputdescr+1;
	}
	else
	  ++i;
      }
    } 


    if(addEOFLine)
    {
      const char *lastLine="EOF\n";

      if(incompleteLineSent)
        watched[0].write("\n",strlen("\n"));

      prepareTimeStamp(0);
      watched[0].write(tstampbuffer,tstampsize);
      watched[0].write(lastLine,strlen(lastLine));
    }
  }

  /**
   * communicate the current CPU time to the time stamper 
   *
   * must only be called from another thread than the one running
   * timeStampLines() !!
   */
  void setCPUtimeFromAnotherThread(float cputime)
  {
    pthread_mutex_lock(&cputimeMutex);
    lastKnownCPUtime=cputime;
    pthread_mutex_unlock(&cputimeMutex);
  }

 protected:

  /**
   * get the current time and store the ascii representation of the
   * time stamp in tstampbuffer (tstampsize will contain the number of
   * characters of the representation)
   *
   * @parm  name=one letter name of the watched file (0 if unused)
   */
  void prepareTimeStamp(char name)
  {
    float cputimeCopy;

    getTimeStamp();
    pthread_mutex_lock(&cputimeMutex);
    cputimeCopy=lastKnownCPUtime;
    pthread_mutex_unlock(&cputimeMutex);

    // store time stamp in tstampbuffer
    if(withCPUtime)
    {
      // CPU time+Wall Clock time
      if(name)
	tstampsize=snprintf(tstampbuffer,sizeof(tstampbuffer),
#if WSIZE==32
			    "%c%.2f/%d.%02d\t",
#else
			    "%c%.2f/%ld.%02ld\t",
#endif
			    name,cputimeCopy,tv.tv_sec,tv.tv_usec/10000);
      else
	tstampsize=snprintf(tstampbuffer,sizeof(tstampbuffer),
#if WSIZE==32
			    "%.2f/%d.%02d\t",
#else
			    "%.2f/%ld.%02ld\t",
#endif
			    cputimeCopy,
			    tv.tv_sec,tv.tv_usec/10000);
    }
    else
    {
      // no CPU time
      if(name)
	tstampsize=snprintf(tstampbuffer,sizeof(tstampbuffer),
#if WSIZE==32
			    "%c%d.%02d\t",
#else
			    "%c%ld.%02ld\t",
#endif
			    name,tv.tv_sec,tv.tv_usec/10000);
      else
	tstampsize=snprintf(tstampbuffer,sizeof(tstampbuffer),
#if WSIZE==32
			    "%d.%02d\t",
#else
			    "%ld.%02ld\t",
#endif
			    tv.tv_sec,tv.tv_usec/10000);
    }

  }

 private:
  /**
   * read data available on watched file with index id (in the watched
   * vector) and output the timestamp stream
   * 
   * @return false on EOF
   */
  bool readFrom(int id)
  {
    char buffer[1024];

    int size=read(watched[id].inputdescr,buffer,sizeof(buffer));

    // PTY seem to be special (this is a quick fix that must be cleaned ???)
    if(size<0)
      return false; // ???

    if(size<0)
      throw runtime_error(string("TimeStamper::readFrom(): read failed: ")
			  +strerror(errno));

    if(size==0)
      return false; // indicate EOF

    // create the time stamp once for all the lines we read
    prepareTimeStamp(watched[id].name);

    if(watched[id].EOLBefore)
    {
      watched[id].write(tstampbuffer,tstampsize);
      watched[id].EOLBefore=false;
    }

    char *s=buffer;
    char *eol;

    // split into lines
    while (size>0  && (eol=(char*)memchr(s,'\n',size))!=NULL)
    {
      // output up to EOL included
      watched[id].write(s,eol-s+1);
      size-=eol-s+1;
      s=eol+1;
    
      if(size>0)
	watched[id].write(tstampbuffer,tstampsize);
      else
	watched[id].EOLBefore=true;
    }

    // output the last incomplete line
    if(size>0)
    {
      watched[id].write(s,size);
      incompleteLineSent=true;
    }
    else
      incompleteLineSent=false;

    return true;
  }

private:
  bool withCPUtime; // do we display CPU time in the timestamp ?

  float lastKnownCPUtime; // current CPU time (provided by an external source)
  pthread_mutex_t cputimeMutex; // a mutex to protect access to cputime

  struct timeval tvStart,tv; // first timestamp and current timestamp

  int max; // max(fd)+1 for select
  fd_set watcher; // set of watched file descriptors for use by select

  // buffer that contains the ascii representation of the timestamp
  char tstampbuffer[64]; // a buffer to output the time stamps
  int tstampsize; // size of the timestamp

  bool incompleteLineSent; // true iff the last line we sent didn't have an EOL 

  bool addEOFLine; // true if we must add an 'EOF' line at the end of output

  /**
   * internal information kept on each file to be watched
   */
  struct info
  {
    bool filteredOutput;
    int inputdescr; // file descriptor to watch
    int outputdescr; // file descriptor to use to output the timestamped stream (when filteredOutput is false)
    AbstractFilter *outputfilter; // filter to write to (when filteredOutput is true)
    char name; // a letter that identifies the watched file
    bool EOLBefore; // true iff the first character we read is the start of
                    // a new line

    info(int inputfd, int outputfd, char letter, bool lineStart)
    {
      filteredOutput=false;
      inputdescr=inputfd;
      outputdescr=outputfd;
      name=letter;
      EOLBefore=lineStart;
    }

    info(int inputfd, AbstractFilter *filter, char letter, bool lineStart)
    {
      filteredOutput=true;
      inputdescr=inputfd;
      outputfilter=filter;
      name=letter;
      EOLBefore=lineStart;
    }

    void write(const char *buffer, int len)
    {
      if(filteredOutput)
	outputfilter->write(buffer,len);
      else
	systemWrite(outputdescr,buffer,len);
    }

    /**
     * handle partial writes and EINTR
     */
    void systemWrite(int fd, const char *buf, size_t len)
    {
      const char *p=buf;
      int n;

      do
      {
	n=::write(fd,p,len);
	if(n<0)
	{
	  if(errno==EINTR)
	    continue;

	  perror("write failed: ");
	  break;
	}

	len-=n;
	p+=n;
      }
      while(len);
    }
  };

  vector<info> watched; // list of files we watch
};

#endif
