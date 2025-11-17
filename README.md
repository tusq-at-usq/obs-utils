# Remote imaging toolbox

A collection of tools for remote imaging data aquisition and rel-time processing. 
Constructed as one monorepo with multiple packages, as well as convenience Ubuntu 24.04 PC setup scripts. 
All packages are structured to work together in OOP fashion. 
Multi-threading and ZMQ messaging are used to achieve real-time performance.
Various tools are described below.

## obs_utils

Collection of various miscellaneous utilities to assist with data processing.

## obs_display

Real-time display with overlays

## obs_cameras

Camera interfaces for Alvium, ZWO, and IDS cameras. 
Also camera image aquisition and save classes.

## obs_encoders

Classes to
(1) receive encoder data via serial, save, and broadcast to ZMQ
(2) receive encoder data via ZMQ and send to relevant places

## obs_certus

Similar to obs_encoders, but for Certus IMU.

## obs_controller

Class for real-time control of (Andy's) actuated gimbal, particularl for following a target path.

## obs_tui

Text-based user information screen for displaying real-time system status.

## obs_target

Class to create fast (Jax JIT) functions to display projected target on display and get tracking setpoints

## obs_astro

Plate solve classes and functions (identifying stars and determining true az/el angles) for calibrating gimbals

## obs_cli

Optional command line interface for operating tracking gimbals, displays, and cameras
