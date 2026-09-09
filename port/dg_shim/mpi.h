/*
 * mpi.h — single-process MPI shim for DG (Discontinuous Galerkin) benchmark.
 *
 * DG (MaxwellsGPU3d) uses MPI for multi-GPU/multi-node mesh partitioning
 * (ParMetis), halo exchange, and global reductions.  For functional/GPGPU-Sim
 * testing we run a single rank (nprocs = 1, procid = 0), under which every MPI
 * call degenerates to either a no-op or a local self-copy:
 *   - Init/Finalize/Barrier/Comm_dup : no-op
 *   - Comm_rank / Comm_size          : 0 / 1
 *   - Allreduce/Reduce/Allgather/
 *     Alltoall/Alltoallv             : single rank => memcpy local buffer
 *   - Isend/Irecv/Waitall            : self-to-self transfer matched by tag
 *
 * This is a header-only replacement for <mpi.h>; place its directory first on
 * the include path (-I port/dg_shim) so it shadows the real MPI header.
 * It is deliberately self-contained (no external MPI library required).
 */
#ifndef DG_MPI_SHIM_H
#define DG_MPI_SHIM_H

#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque / scalar types ---- */
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;

typedef struct MPI_Status {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
  int _count;
} MPI_Status;

/* ---- Communicators ---- */
#define MPI_COMM_WORLD 0
#define MPI_COMM_SELF  1

/* ---- Datatypes (values are arbitrary but unique) ---- */
#define MPI_CHAR    1
#define MPI_SHORT   2
#define MPI_INT     3
#define MPI_LONG    4
#define MPI_FLOAT   5
#define MPI_DOUBLE  6
#define MPI_UNSIGNED 7

/* ---- Reduction operators (no-ops with a single rank) ---- */
#define MPI_MIN 10
#define MPI_MAX 11
#define MPI_SUM 12
#define MPI_PROD 13

/* ---- Return codes ---- */
#define MPI_SUCCESS 0

/* Special "in place" buffer alias for collectives (rarely used). */
#ifndef MPI_IN_PLACE
#define MPI_IN_PLACE ((void *)0x1)
#endif

static inline int mpi_shim_type_size(MPI_Datatype t) {
  switch (t) {
    case MPI_CHAR:     return (int)sizeof(char);
    case MPI_SHORT:    return (int)sizeof(short);
    case MPI_INT:      return (int)sizeof(int);
    case MPI_LONG:     return (int)sizeof(long);
    case MPI_FLOAT:    return (int)sizeof(float);
    case MPI_DOUBLE:   return (int)sizeof(double);
    case MPI_UNSIGNED: return (int)sizeof(unsigned int);
    default:           return 1;
  }
}

/* ---- Lifecycle / communicator queries ---- */
static inline int MPI_Init(int *argc, char ***argv) {
  (void)argc; (void)argv; return MPI_SUCCESS;
}
static inline int MPI_Finalize(void) { return MPI_SUCCESS; }
static inline int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  (void)comm; *rank = 0; return MPI_SUCCESS;
}
static inline int MPI_Comm_size(MPI_Comm comm, int *size) {
  (void)comm; *size = 1; return MPI_SUCCESS;
}
static inline int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  *newcomm = comm; return MPI_SUCCESS;
}
static inline int MPI_Barrier(MPI_Comm comm) { (void)comm; return MPI_SUCCESS; }
static inline double MPI_Wtime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * Point-to-point request registry.  Every request on a single rank is a
 * self-to-self transfer; Irecv/Isend record the buffer, and Waitall matches
 * receives to sends by tag and performs the copy.  All three calls for a given
 * exchange live in the same translation unit (LoadBalance3d.c), so a
 * file-local registry is sufficient.
 */
#define MPI_SHIM_MAX_REQ 256
typedef struct {
  void *buf;
  size_t bytes;
  int tag;
  int is_send;
  int active;
} mpi_shim_req_t;

static inline mpi_shim_req_t *mpi_shim_registry(void) {
  static mpi_shim_req_t reg[MPI_SHIM_MAX_REQ];
  static int inited = 0;
  if (!inited) { memset(reg, 0, sizeof(reg)); inited = 1; }
  return reg;
}

static inline int MPI_Irecv(void *buf, int count, MPI_Datatype dtype,
                            int src, int tag, MPI_Comm comm,
                            MPI_Request *request) {
  (void)src; (void)comm;
  mpi_shim_req_t *reg = mpi_shim_registry();
  int slot;
  for (slot = 0; slot < MPI_SHIM_MAX_REQ; ++slot)
    if (!reg[slot].active) break;
  reg[slot].buf = buf;
  reg[slot].bytes = (size_t)count * (size_t)mpi_shim_type_size(dtype);
  reg[slot].tag = tag;
  reg[slot].is_send = 0;
  reg[slot].active = 1;
  *request = slot;
  return MPI_SUCCESS;
}

static inline int MPI_Isend(const void *buf, int count, MPI_Datatype dtype,
                            int dst, int tag, MPI_Comm comm,
                            MPI_Request *request) {
  (void)dst; (void)comm;
  mpi_shim_req_t *reg = mpi_shim_registry();
  int slot;
  for (slot = 0; slot < MPI_SHIM_MAX_REQ; ++slot)
    if (!reg[slot].active) break;
  reg[slot].buf = (void *)buf;
  reg[slot].bytes = (size_t)count * (size_t)mpi_shim_type_size(dtype);
  reg[slot].tag = tag;
  reg[slot].is_send = 1;
  reg[slot].active = 1;
  *request = slot;
  return MPI_SUCCESS;
}

static inline int MPI_Waitall(int count, MPI_Request *requests,
                              MPI_Status *statuses) {
  mpi_shim_req_t *reg = mpi_shim_registry();
  int i;
  for (i = 0; i < count; ++i) {
    mpi_shim_req_t *r = &reg[requests[i]];
    if (!r->active) continue;
    if (!r->is_send) {
      /* Match this receive to an active send with the same tag. */
      int j;
      for (j = 0; j < MPI_SHIM_MAX_REQ; ++j) {
        if (reg[j].active && reg[j].is_send && reg[j].tag == r->tag) {
          size_t n = r->bytes < reg[j].bytes ? r->bytes : reg[j].bytes;
          memcpy(r->buf, reg[j].buf, n);
          reg[j].active = 0;
          break;
        }
      }
      if (statuses) { statuses[i].MPI_TAG = r->tag; statuses[i]._count = (int)r->bytes; }
    }
    r->active = 0;
  }
  return MPI_SUCCESS;
}

/* ---- Collectives: single rank => local copy ---- */
static inline int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  (void)op; (void)comm;
  size_t bytes = (size_t)count * (size_t)mpi_shim_type_size(dtype);
  if (sendbuf != MPI_IN_PLACE && sendbuf != recvbuf)
    memcpy(recvbuf, sendbuf, bytes);
  return MPI_SUCCESS;
}

static inline int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype dtype, MPI_Op op, int root,
                             MPI_Comm comm) {
  (void)op; (void)root; (void)comm;
  size_t bytes = (size_t)count * (size_t)mpi_shim_type_size(dtype);
  if (sendbuf != recvbuf) memcpy(recvbuf, sendbuf, bytes);
  return MPI_SUCCESS;
}

static inline int MPI_Allgather(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm) {
  (void)recvcount; (void)comm;
  size_t bytes = (size_t)sendcount * (size_t)mpi_shim_type_size(sendtype);
  (void)recvtype;
  if (sendbuf != recvbuf) memcpy(recvbuf, sendbuf, bytes);
  return MPI_SUCCESS;
}

static inline int MPI_Alltoall(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm) {
  (void)comm; (void)recvcount; (void)recvtype;
  size_t bytes = (size_t)sendcount * (size_t)mpi_shim_type_size(sendtype);
  if (sendbuf != recvbuf) memcpy(recvbuf, sendbuf, bytes);
  return MPI_SUCCESS;
}

static inline int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, MPI_Datatype sendtype,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, MPI_Datatype recvtype,
                                MPI_Comm comm) {
  (void)comm; (void)recvtype;
  int esz = mpi_shim_type_size(sendtype);
  /* Single rank: move rank-0 send chunk to rank-0 recv slot. */
  size_t nbytes = (size_t)recvcounts[0] * (size_t)esz;
  memcpy((char *)recvbuf + (size_t)rdispls[0] * (size_t)esz,
         (const char *)sendbuf + (size_t)sdispls[0] * (size_t)esz,
         nbytes);
  return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif /* DG_MPI_SHIM_H */
