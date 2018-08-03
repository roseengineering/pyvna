pyvna
-------------------

Enclosed is a Python3 library for performing Vector Network Analyzer 
measurements using various computer controlled VNAs.  The 
library is intended to be with Jupyter Notebook.  The library supports SOLT calibration as well as plotting.

Please see the sample Jupyter Notebooks in the repo for examples
of use.

Devices
---------------------

At the moment only the RigExpert AA-30,
and MiniVNA Tiny devices are supported.  The NWT500 is also supported
but it untested.  Each device has an associated driver.
For example the Zero's driver is located in device\_zero.py.
If you want to create a driver please do.


Getting Started
-------------------

After initializing the VNA using,

     %matplotlib inline
     import pyvna as vna
     vna.init(port='/dev/ttyUSB0', name='zero')

the calibration object must be initialized.  The
calibration object is given the start frequency, the
stop frequency, and the number of points to sweep.
By default it will start from the lowest frequency possible 
for the device and sweep
to its highest.

     cal = vna.create()

Next you need to calibrate the VNA.  At the moment only
SOLT calibration is supported.  If you want to measure
S11 you need to calibrate against a open, short, and load
standard.  This is done by passing the created calibration object
to the cal\_open(), cal\_short(), and cal\_load() methods.  For example:

     gm = vna.cal_open(cal)
     gm = vna.cal_short(cal)
     gm = vna.cal_load(cal)

All these methods return the raw gamma values from the VNA 
which can be ploted in Jupyter Notebook with the helper
methods plot(), bode(), cartesian(), and polar().  For example:

     vna.polar(gm)
     vna.bode(gm)

To run the return loss measurement use the method return\_loss()
passing the calibration object.

     gm = vna.return_loss(cal)
     vna.polar(gm)
     vna.bode(gm)


Other features
--------------

The library also supports thru, thrumatch, crosstalk calibration.
In addition the library has the ability to measure in real time
response (S21), enhanced response (S21 off a prior S11M() measurement
run), and a forward path measurement (S11 and S21).
A full two port measurement is supported by running the four measurements,
S11M(), S21M(), S21M(), and S22M().  Calling the method 
two\_port() afterwards will return the corrected two port results.

Unfortunately these features are relatively untested at the moment.  

Configuration
-------------

Normally the device to use is set using the init() method as
shown above.

But as a convenience, the default device to use can be set in the 
configuration file, device.conf.
You do this by setting the 'name' option under the [device] heading.
Options specific to a particular device are set under that
device's name as a heading.  


Requirements
------------

The library depends on the Python3 packages numpy and pandas to be installed 
as well as the package pyserial if needed by the device.  The package 
matplotlib is required if plotting is used.




