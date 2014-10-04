from functions import *

print "Attempting to read cached updates data."
file = open("sample_times,samples,period,modes,updates.pickle", "rb")
if file:
	sample_times, samples, period, modes, updates = UnpickleThis("sample_times,samples,period,modes,updates.pickle")
	print "Read in cached updates data.  There are {0} updates.".format(len(updates))
else:
	print "No cached updates data.  Starting fresh from dynamics data."

	# sample_times is a list of scalars, samples is a list of 3-vectors, and period is a positive scalar.
	# The Heisenberg dynamics as defined by the data in this file is defined as follows.  The elements
	# in samples and sample_times (of which there should be an identical number) pair up to form pairs
	# of the form (sample_times[0],samples[0]), (sample_times[1],samples[1]), ..., which define a
	# discrete sampling (t,p(t)) of the path function p : [0,period] -> R^3.
	print "Attempting to read dynamics data."
	sample_times, samples, period = UnpickleThis("sample_times,samples,period.pickle")
	# Turn the list of 3-vector samples into a list of complex 3-vectors so that we can pipe it through
	# the Fourier code more readily.
	samples = [ComplexVector(*sample) for sample in samples]
	print "period = {0}, len(sample_times) = {1}, len(samples) = {2}".format(period, len(sample_times), len(samples))

	# L is the number (on each side of 0) of coefficients to use in the Fourier sum.
	L = 30
	# modes is the list of modes (cycle counts based on the period) to use in Fourier operations.
	# We can have this be a sparser list, for example [-9, -6, -3, 0, 3, 6, 9], in which case,
	# the missing modes are assumed to have Fourier coefficient 0.
	modes = [k for k in range(-L,L+1)]
	# Compute the Fourier coefficients of the Heisenberg dynamics that was read in from the file.
	print "Computing the Fourier coefficients of the dynamics."
	coefficients = FourierExpansion(sample_times, samples, period, modes)
	# Initialize the updates list with the initial set of Fourier coefficients.
	updates = [coefficients]

# This Sage pane runs step_count iterations of the [gradient descent] Update function.
# The higher step_count is, the longer it will take (the time will increase linearly
# with step_count).  Note that RunUpdates stores its computations in the updates list,
# and if you run RunUpdates again, it will continue where it left off.  Thus in theory
# it should be possible to interrupt (Esc key) the process and then continue it later
# without losing data.

step_count = 400
step_size = 0.2
print "Computing {0} gradient descent updates.".format(step_count)
RunUpdates(modes, period, updates, step_count, step_size)
print "There are now {0} updates computed.  Caching this data to file.".format(len(updates))
data = [sample_times, samples, period, modes, updates]
PickleThis(data, "sample_times,samples,period,modes,updates.pickle")

# This reconstructs the paths from the Fourier coefficients in updates, and stores them in a file.

print "Reconstructing the dynamics from its Fourier coefficients (for each update)."
sparser_sample_times = [t for t in linterp(0.0, 273.5, value_count=400)]
# reconstructed_samples = [[FourierSum(modes, coefficients, period, t) for t in sparser_sample_times] for coefficients in updates]
reconstructed_samples = []
for (i,coefficients) in enumerate(updates):
	print "Reconstructing dynamics for update {0}.".format(i)
	reconstructed_samples.append([FourierSum(modes, coefficients, period, t) for t in sparser_sample_times])
print "Storing reconstructed dynamics to file."
PickleThis(reconstructed_samples, "reconstructed_samples.pickle")
