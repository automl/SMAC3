VERSION=3.3.4
#SVNVERSION=`svnversion .`
SVNVERSION=$(word 2,$$Rev: 1649 $$)
#DEBUG=-g
DEBUG=-O3

STATIC=
#STATIC=-static

WSIZE:=$(shell if [ `uname -m` = 'x86_64' ] ; then echo 64; else echo 32; fi )
SYSCALLINCFILE:=$(shell  for f in /usr/include/asm/unistd_$(WSIZE).h /usr/include/`uname -m`-linux-gnu/asm/unistd_$(WSIZE).h ; do if [ -e $$f ] ; then echo $$f ; break ; fi ; done )

CFLAGS=-Dtmpdebug -Wall -DVERSION=\"$(VERSION)\" -DSVNVERSION=\"$(SVNVERSION)\" -DWSIZE=$(WSIZE)

SRC=runsolver.cc SignalNames.cc SyscallNames.cc
OBJ=$(SRC:.cc=.o)

all:runsolver

install: runsolver
	cp runsolver $(INSTROOT)/usr/bin

include $(SRC:.cc=.d)

.cc.o:
	g++ $(CFLAGS) $(DEBUG) -c $*.cc

runsolver: $(OBJ)
	g++  $(STATIC) $(DEBUG) -o $@ $^ -lpthread

testlimit: testlimit.cc
	g++ -o testlimit testlimit.cc

testthread: testthread.cc
	g++ -o testthread testthread.cc -lpthread

SyscallNames.cc SyscallNames.hh: CreateSyscallsNames.cc $(SYSCALLINCFILE)
	grep '#define[[:space:]]*__NR' $(SYSCALLINCFILE) | grep -v '^/' | awk '{print $$2}' | sed -e 's/^__NR_//' | awk '{printf "list[__NR_%s]=\"%s\";\n",$$1,$$1}' > tmpSyscallList.cc
	g++ -o CreateSyscallsNames CreateSyscallsNames.cc
	./CreateSyscallsNames
	rm tmpSyscallList.cc CreateSyscallsNames 

.PHONY: clean realclean archive

tar: /tmp/runsolver-$(VERSION).tar.bz2
archive: /tmp/runsolver-$(VERSION).tar.bz2

/tmp/runsolver-$(VERSION).tar.bz2: realclean $(SRC) Makefile
	sed -i -e 's/^Version:.*/Version:\t'$(VERSION)'/' runsolver.spec
	tar cvjf /tmp/runsolver-$(VERSION).tar.bz2 -C ../.. runsolver/src --exclude .svn

rpm: /tmp/runsolver-$(VERSION).tar.bz2
	rpmbuild -tb /tmp/runsolver-$(VERSION).tar.bz2

srpm: /tmp/runsolver-$(VERSION).tar.bz2
	rpmbuild -ts /tmp/runsolver-$(VERSION).tar.bz2

clean:
	rm -f runsolver $(OBJ) *.class testlimit testtimestamper vlineSplitter testProcessTree runtestlimit testthread

realclean: clean
	rm -f *.d *~ SyscallNames.*


%.d: %.cc
	$(SHELL) -ec '$(CC) -MM $(CFLAGS) $< \
	| sed -e '\''s/$*\.o[ :]*/$@ &/g'\'' > $@'


