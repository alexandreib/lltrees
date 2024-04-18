CC=g++ -g
SOURCEDIR = cpp
BUILDDIR = build

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))
TARGETS = $(BUILDDIR)/lltrees.so 

BOOST = -DBOOST_BIND_GLOBAL_PLACEHOLDERS -DBOOST_ALLOW_DEPRECATED_HEADERS
PYLIBPATH=$(shell python3-config --exec-prefix)/lib
PY_CFLAGS=$(shell python3-config --cflags) -fPIC -c
PY_LDFLAGS=$(shell python3-config --ldflags) -shared -Wl,-rpath,$(PYLIBPATH) 

py_version?=310
ifeq ($(py_version), 38)
boost_version=-lpython3.8 -lboost_python38 -lboost_numpy38	
else ifeq ($(py_version), 310)
boost_version=-lpython3.10 -lboost_python310 -lboost_numpy310	
endif

#pp:
#	@echo $(boost_version)
#	@echo $(py_version)

all: build
 

build: $(TARGETS)

$(TARGETS) : $(OBJECTS)
	$(CC) -o $@ $^ $(PY_LDFLAGS) $(boost_version)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp
	$(CC) -o $@ $^ $(PY_CFLAGS) -std=c++17 $(BOOST)

clean:
	rm -rf $(BUILDDIR)
	mkdir -p $(BUILDDIR)

print-%  : ; @echo $* = $($*)
.PHONY: default clean
