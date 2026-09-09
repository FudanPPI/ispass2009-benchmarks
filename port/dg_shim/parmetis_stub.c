/*
 * parmetis_stub.c — single-rank stub for the one ParMETIS entry point used by
 * the DG benchmark's host-side mesh load balancer (LoadBalance3d.c).
 *
 * With nprocs = 1 every mesh element already resides on rank 0, so the mesh
 * k-way partition is trivial: assign every element to partition 0.  The
 * subsequent self-directed MPI_Isend/Irecv exchange in LoadBalance3d() then
 * reconstructs identical arrays, making the whole redistribution a no-op.
 *
 * idxtype and MPI_Comm come from DG's include/parmetis.h, which in turn
 * includes the single-process mpi.h shim.
 */
#include <parmetis.h>
#include <stdlib.h>
#include <string.h>

/*
 * METIS/ParMetIS renamed memory helpers.  rename.h maps the public names
 * (idxmalloc, imalloc, fmalloc, ...) to double-underscore symbols that are
 * normally provided by libmetis/libparmetis.  In a normal build these are
 * thin malloc wrappers; we provide them here so the single-rank host code
 * links without the libraries.
 */
idxtype *idxmalloc__(int n, char *msg) {
  (void)msg;
  return (idxtype *)malloc(sizeof(idxtype) * (size_t)n);
}
int *imalloc__(int n, char *msg) {
  (void)msg;
  return (int *)malloc(sizeof(int) * (size_t)n);
}
float *fmalloc__(int n, char *msg) {
  (void)msg;
  return (float *)malloc(sizeof(float) * (size_t)n);
}
int *ismalloc__(int n, int val, char *msg) {
  int *p = imalloc__(n, msg);
  int i;
  for (i = 0; i < n; ++i) p[i] = val;
  return p;
}
idxtype *idxsmalloc__(int n, idxtype val, char *msg) {
  idxtype *p = idxmalloc__(n, msg);
  int i;
  for (i = 0; i < n; ++i) p[i] = val;
  return p;
}
void *GKmalloc__(size_t n, char *msg) {
  (void)msg;
  return malloc(n);
}

void __cdecl ParMETIS_V3_PartMeshKway(
    idxtype *elmdist, idxtype *eptr, idxtype *eind, idxtype *elmwgt,
    int *wgtflag, int *numflag, int *ncon, int *ncommonnodes, int *nparts,
    float *tpwgts, float *ubvec, int *options, int *edgecut, idxtype *part,
    MPI_Comm *comm) {
  (void)eptr; (void)eind; (void)elmwgt;
  (void)wgtflag; (void)numflag; (void)ncon; (void)ncommonnodes; (void)nparts;
  (void)tpwgts; (void)ubvec; (void)options; (void)comm;

  /* Rank 0 owns elmdist[0]..elmdist[1] local elements; all go to partition 0. */
  int nlocal = (int)(elmdist[1] - elmdist[0]);
  int i;
  for (i = 0; i < nlocal; ++i)
    part[i] = 0;

  if (edgecut)
    *edgecut = 0;
}
