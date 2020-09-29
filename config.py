import odl
import torch.nn as nn
import torch
import numpy as np
import sys, termios, tty, os, time
import fcntl

import warnings

warnings.filterwarnings("ignore")##Ignore warnings

## Path where the data is stored. Has the following structure
#
# ├── data
# │   ├── ELLIP
# │   ├── LIDC_frmsubset
# │   ├── MAYO
# │   ├── MNIST
# │   └── STL10
# ├── figs
# │   ├── ADR2
# │   ├── ADR4
# │   ├── ADR5
# │   ├── CLAR04
# │   ├── CLAR05
# │   ├── CLAR4
# │   ├── CLAR5
# │   ├── compare
# │   ├── compare5
# │   ├── FBP+U4
# │   ├── LPD4
# │   ├── LPD5
# │   └── TV4
# ├── logs
# │   ├── ADR
# │   ├── CLAR
# │   ├── CLAR0
# │   ├── GLOW
# │   └── LPD
# └── nets
#     ├── net_compare ---- network checkpoints for comparison
#     ├── net_cps ---- directory to save all network checkpoints
#     └── netCLAR5.pt ---- checkpoint to be used by train.py
#


##
## Function for receiving input from keyboard
##
def getch():
  fd = sys.stdin.fileno()

  oldterm = termios.tcgetattr(fd)
  newattr = termios.tcgetattr(fd)
  newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
  termios.tcsetattr(fd, termios.TCSANOW, newattr)

  oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
  fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

  try:
    while 1:
      try:
        c = sys.stdin.read(1)
        break
      except IOError: pass
  finally:
    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
  return c



##
## Making sure that algortihms can be loaded from the Algorithms folder
##
sys.path.append('./Algorithms/')


##
## Have to redefine the operator module term, as otherwise gradient calculation is wrong
## Copied from the newest version of odl
##

from odl import Operator
class OperatorFunction(torch.autograd.Function):

    """Wrapper of an ODL operator as a ``torch.autograd.Function``.
    """

    @staticmethod
    def forward(ctx, operator, input):
        """Evaluate forward pass on the input.
        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        operator : `Operator`
            ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.
        input : `torch.Tensor`
            Point at which to evaluate the operator.
        Returns
        -------
        result : `torch.Tensor`
            Tensor holding the result of the evaluation.
        """
        if not isinstance(operator, Operator):
            raise TypeError(
                "`operator` must be an `Operator` instance, got {!r}"
                "".format(operator)
            )

        # Save operator for backward; input only needs to be saved if
        # the operator is nonlinear (for `operator.derivative(input)`)
        ctx.operator = operator

        if not operator.is_linear:
            # Only needed for nonlinear operators
            ctx.save_for_backward(input)

        # TODO(kohr-h): use GPU memory directly when possible
        # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
        # is required
        input_arr = copy_if_zero_strides(input.cpu().detach().numpy())

        # Determine how to loop over extra shape "left" of the operator
        # domain shape
        in_shape = input_arr.shape
        op_in_shape = operator.domain.shape
        if operator.is_functional:
            op_out_shape = ()
            op_out_dtype = operator.domain.dtype
        else:
            op_out_shape = operator.range.shape
            op_out_dtype = operator.range.dtype

        extra_shape = in_shape[:-len(op_in_shape)]
        if in_shape[-len(op_in_shape):] != op_in_shape:
            shp_str = str(op_in_shape).strip('(,)')
            raise ValueError(
                'input tensor has wrong shape: expected (*, {}), got {}'
                ''.format(shp_str, in_shape)
            )

        # Store some information on the context object
        ctx.op_in_shape = op_in_shape
        ctx.op_out_shape = op_out_shape
        ctx.extra_shape = extra_shape
        ctx.op_in_dtype = operator.domain.dtype
        ctx.op_out_dtype = op_out_dtype

        # Evaluate the operator on all inputs in a loop
        if extra_shape:
            # Multiple inputs: flatten extra axes, then do one entry at a time
            input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
            results = []
            for inp in input_arr_flat_extra:
                results.append(operator(inp))

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_out_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_out_shape)
        else:
            # Single input: evaluate directly
            result_arr = np.asarray(
                operator(input_arr)
            ).astype(op_out_dtype, copy=False)

        # Convert back to tensor
        tensor = torch.from_numpy(result_arr).to(input.device)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):

        # Return early if there's nothing to do
        if not ctx.needs_input_grad[1]:
            return None, None

        operator = ctx.operator

        # Get `operator` and `input` from the context object (the input
        # is only needed for nonlinear operators)
        if not operator.is_linear:
            # TODO: implement directly for GPU data
            # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
            # is required
            input_arr = copy_if_zero_strides(
                ctx.saved_tensors[0].detach().cpu().numpy()
            )

        # ODL weights spaces, pytorch doesn't, so we need to handle this
        try:
            dom_weight = operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0
        try:
            ran_weight = operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0
        scaling = dom_weight / ran_weight

        # Convert `grad_output` to NumPy array
        grad_output_arr = copy_if_zero_strides(
            grad_output.detach().cpu().numpy()
        )

        # Get shape information from the context object
        op_in_shape = ctx.op_in_shape
        op_out_shape = ctx.op_out_shape
        extra_shape = ctx.extra_shape
        op_in_dtype = ctx.op_in_dtype

        # Check if `grad_output` is consistent with `extra_shape` and
        # `op_out_shape`
        if grad_output_arr.shape != extra_shape + op_out_shape:
            raise ValueError(
                'expected tensor of shape {}, got shape {}'
                ''.format(extra_shape + op_out_shape, grad_output_arr.shape)
            )

        # Evaluate the (derivative) adjoint on all inputs in a loop
        if extra_shape:
            # Multiple gradients: flatten extra axes, then do one entry
            # at a time
            grad_output_arr_flat_extra = grad_output_arr.reshape(
                (-1,) + op_out_shape
            )

            results = []
            if operator.is_linear:
                for ograd in grad_output_arr_flat_extra:
                    results.append(np.asarray(operator.adjoint(ograd)))
            else:
                # Need inputs, flattened in the same way as the gradients
                input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
                for ograd, inp in zip(
                    grad_output_arr_flat_extra, input_arr_flat_extra
                ):
                    results.append(
                        np.asarray(operator.derivative(inp).adjoint(ograd))
                    )

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_in_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_in_shape)
        else:
            # Single gradient: evaluate directly
            if operator.is_linear:
                result_arr = np.asarray(
                    operator.adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)
            else:
                result_arr = np.asarray(
                    operator.derivative(input_arr).adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)

        # Apply scaling, convert to tensor and return
        if scaling != 1.0:
            result_arr *= scaling
        grad_input = torch.from_numpy(result_arr).to(grad_output.device)
        return None, grad_input  # return `None` for the `operator` part
class OperatorModule(torch.nn.Module):

    """Wrapper of an ODL operator as a ``torch.nn.Module``.
    """

    def __init__(self, operator):
        """Initialize a new instance."""
        super(OperatorModule, self).__init__()
        self.operator = operator

    def forward(self, x):
        """Compute forward-pass of this module on ``x``.
        """
        in_shape = tuple(x.shape)
        in_ndim = len(in_shape)
        op_in_shape = self.operator.domain.shape
        op_in_ndim = len(op_in_shape)
        if in_ndim <= op_in_ndim or in_shape[-op_in_ndim:] != op_in_shape:
            shp_str = str(op_in_shape).strip('()')
            raise ValueError(
                'input tensor has wrong shape: expected (N, *, {}), got {}'
                ''.format(shp_str, in_shape)
            )
        return OperatorFunction.apply(self.operator, x)

    def __repr__(self):
        """Return ``repr(self)``."""
        op_name = self.operator.__class__.__name__
        op_in_shape = self.operator.domain.shape
        if len(op_in_shape) == 1:
            op_in_shape = op_in_shape[0]
        op_out_shape = self.operator.range.shape
        if len(op_out_shape) == 1:
            op_out_shape = op_out_shape[0]

        return '{}({}) ({} -> {})'.format(
            self.__class__.__name__, op_name, op_in_shape, op_out_shape
        )
def copy_if_zero_strides(arr):
    """Workaround for NumPy issue #9165 with 0 in arr.strides."""
    assert isinstance(arr, np.ndarray)
    return arr.copy() if 0 in arr.strides else arr



##
## Initialise operators and the data parameters
##
def init(args):
    global fwd_op_mod, fbp_op_mod, fwd_op_adj_mod
    global size, angles, noise, rec_size,setup,detectors
    global space, data_path,fwd_op

    noise = args.noise
    setup = args.setup
    max_ang=np.pi#by default we are dealing with normal CT, not limited angle, for limted angle this value is changed
    ##
    ## Setup 1 - 28x28 images from MNIST
    ##
    detectors=args.detectors
    data_path = args.data_path
    if(args.setup == 1):
        size = 32
        angles = 5
        args.data_path = args.data_path + 'data/MNIST'

    ##
    ## Setup 2 - 128x128 ellipses images from dival
    ##
    elif(args.setup == 2):
        size = 128
        angles = 128
        args.data_path = args.data_path + 'data/ELLIP'
    ##
    ## Setup 3 - 96x96 STL10 images to deoise
    ##
    elif(args.setup == 3):
        size = 96
        angles = 0
        args.data_path = args.data_path + 'data/STL10'

    ##
    ## Setup 4 - 512x512 LIDC images from dival
    ##
    elif(args.setup == 4):
        size = 512
        angles = 1000
        detectors=513
        args.data_path = args.data_path + 'data/LIDC_frmsubset'


    ##
    ## Setup 5 - 512x512 Mayo images
    ##
    elif(args.setup == 5):
        size = 512
        angles = 350
        max_ang=2*np.pi/3

        args.data_path = args.data_path + 'data/MAYO'



    space = odl.uniform_discr([-size//2, -size//2], [size//2, size//2], [size, size], dtype='float32')
    if angles != 0:
        angle_partition = odl.uniform_partition(0,max_ang, angles)
        detector_partition = odl.uniform_partition(-362.0, 362.0, args.detectors)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)
        # geometry = odl.tomo.parallel_beam_geometry(space, num_angles=angles, det_shape=(detectors,))
        fwd_op = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')


        fbp_op = odl.tomo.fbp_op(fwd_op)#,filter_type='Hann')
        fwd_op_mod=OperatorModule(fwd_op)
        fwd_op_adj_mod=OperatorModule(fwd_op.adjoint)

        rec_size = fwd_op.range.shape[1]
        fbp_op_mod = OperatorModule(fbp_op)
