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



#include "SignalNames.hh"

const char *signalNames[]={
  "???",
  "SIGHUP",         
  "SIGINT",        
  "SIGQUIT",        
  "SIGILL",         
  "SIGTRAP",
  "SIGABRT",         
  "SIGBUS",          
  "SIGFPE",          
  "SIGKILL",         
  "SIGUSR1",         
  "SIGSEGV",         
  "SIGUSR2",         
  "SIGPIPE",         
  "SIGALRM",         
  "SIGTERM",         
  "SIGSTKFLT",       
  "SIGCHLD",         
  "SIGCONT",         
  "SIGSTOP",         
  "SIGTSTP",         
  "SIGTTIN",         
  "SIGTTOU",         
  "SIGURG",          
  "SIGXCPU",         
  "SIGXFSZ",         
  "SIGVTALRM",       
  "SIGPROF",         
  "SIGWINCH",        
  "SIGIO",           
  "SIGPWR",          
  "SIGSYS"};


const char *getSignalName(int sig)
{
  if (sig>0 && sig<=static_cast<int>(sizeof(signalNames)/sizeof(char *)))
    return signalNames[sig];
  else
    return "???";
}
