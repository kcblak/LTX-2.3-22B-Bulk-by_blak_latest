# Architecture Guide

The system is organized as a headless rendering platform with clear separation between configuration, validation, rendering, synchronization, stitching, observability, and reporting.

## Pipeline
1. Bootstrap
2. Layered configuration load
3. Diagnostics and preflight
4. Job queue initialization and resume
5. Renderer initialization
6. Sequential rendering with background uploads
7. Optional final stitching
8. Artifact reporting and clean shutdown
