/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/


#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TBB )

#include <cstdint>
#include <limits>
#include <utility>
#include <iostream>
#include <sstream>

#include <Kokkos_Core.hpp>

#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_CPUDiscovery.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {
namespace {

TBBExec                      s_tbb_process ;
TBBExec                    * s_tbb_exec[  TBBExec::MAX_THREAD_COUNT ] = { 0 };
pthread_t                    s_tbb_pid[   TBBExec::MAX_THREAD_COUNT ] = { 0 };
std::pair<unsigned,unsigned> s_tbb_coord[ TBBExec::MAX_THREAD_COUNT ];

int s_tbb_pool_size[3] = { 0 , 0 , 0 };

unsigned s_current_reduce_size = 0 ;
unsigned s_current_shared_size = 0 ;

void (* volatile s_current_function)( TBBExec & , const void * );
const void * volatile s_current_function_arg = 0 ;

struct Sentinel {
  Sentinel()
  {}

  ~Sentinel()
  {
    if ( s_tbb_pool_size[0] ||
         s_tbb_pool_size[1] ||
         s_tbb_pool_size[2] ||
         s_current_reduce_size ||
         s_current_shared_size ||
         s_current_function ||
         s_current_function_arg ||
         s_tbb_exec[0] ) {
      std::cerr << "ERROR : Process exiting without calling Kokkos::TBB::terminate()" << std::endl ;
    }
  }
};

inline
unsigned fan_size( const unsigned rank , const unsigned size )
{
  const unsigned rank_rev = size - ( rank + 1 );
  unsigned count = 0 ;
  for ( unsigned n = 1 ; ( rank_rev + n < size ) && ! ( rank_rev & n ) ; n <<= 1 ) { ++count ; }
  return count ;
}

} // namespace
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

void execute_function_noop( TBBExec & , const void * ) {}

void TBBExec::driver(void)
{
  SharedAllocationRecord< void, void >::tracking_enable();

  TBBExec this_thread ;

  while ( TBBExec::Active == this_thread.m_pool_state ) {

    (*s_current_function)( this_thread , s_current_function_arg );

    // Deactivate thread and wait for reactivation
    this_thread.m_pool_state = TBBExec::Inactive ;

    wait_yield( this_thread.m_pool_state , TBBExec::Inactive );
  }
}

TBBExec::TBBExec()
  : m_pool_base(0)
  , m_scratch(0)
  , m_scratch_reduce_end(0)
  , m_scratch_tbb_end(0)
  , m_numa_rank(0)
  , m_numa_core_rank(0)
  , m_pool_rank(0)
  , m_pool_size(0)
  , m_pool_fan_size(0)
  , m_pool_state( TBBExec::Terminating )
{
  if ( & s_tbb_process != this ) {

    // A spawned thread

    TBBExec * const nil = 0 ;

    // Which entry in 's_tbb_exec', possibly determined from hwloc binding
    const int entry = ((size_t)s_current_function_arg) < size_t(s_tbb_pool_size[0])
                    ? ((size_t)s_current_function_arg)
                    : size_t(Kokkos::hwloc::bind_this_thread( s_tbb_pool_size[0] , s_tbb_coord ));

    // Given a good entry set this thread in the 's_tbb_exec' array
    if ( entry < s_tbb_pool_size[0] &&
         nil == atomic_compare_exchange( s_tbb_exec + entry , nil , this ) ) {

      const std::pair<unsigned,unsigned> coord = Kokkos::hwloc::get_this_tbb_coordinate();

      m_numa_rank       = coord.first ;
      m_numa_core_rank  = coord.second ;
      m_pool_base       = s_tbb_exec ;
      m_pool_rank       = s_tbb_pool_size[0] - ( entry + 1 );
      m_pool_rank_rev   = s_tbb_pool_size[0] - ( pool_rank() + 1 );
      m_pool_size       = s_tbb_pool_size[0] ;
      m_pool_fan_size   = fan_size( m_pool_rank , m_pool_size );
      m_pool_state      = TBBExec::Active ;

      s_tbb_pid[ m_pool_rank ] = pthread_self();

      // Inform spawning process that the threads_exec entry has been set.
      s_tbb_process.m_pool_state = TBBExec::Active ;
    }
    else {
      // Inform spawning process that the threads_exec entry could not be set.
      s_tbb_process.m_pool_state = TBBExec::Terminating ;
    }
  }
  else {
    // Enables 'parallel_for' to execute on unitialized Threads device
    m_pool_rank  = 0 ;
    m_pool_size  = 1 ;
    m_pool_state = TBBExec::Inactive ;

    s_tbb_pid[ m_pool_rank ] = pthread_self();
  }
}

TBBExec::~TBBExec()
{
  const unsigned entry = m_pool_size - ( m_pool_rank + 1 );

  typedef Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::HostSpace , void > Record ;

  if ( m_scratch ) {
    Record * const r = Record::get_record( m_scratch );

    m_scratch = 0 ;

    Record::decrement( r );
  }

  m_pool_base   = 0 ;
  m_scratch_reduce_end = 0 ;
  m_scratch_tbb_end = 0 ;
  m_numa_rank      = 0 ;
  m_numa_core_rank = 0 ;
  m_pool_rank      = 0 ;
  m_pool_size      = 0 ;
  m_pool_fan_size  = 0 ;

  m_pool_state  = TBBExec::Terminating ;

  if ( & s_tbb_process != this && entry < MAX_THREAD_COUNT ) {
    TBBExec * const nil = 0 ;

    atomic_compare_exchange( s_tbb_exec + entry , this , nil );

    s_tbb_process.m_pool_state = TBBExec::Terminating ;
  }
}


int TBBExec::get_tbb_count()
{
  return s_tbb_pool_size[0] ;
}

TBBExec * TBBExec::get_thread( const int init_tbb_rank )
{
  TBBExec * const th =
    init_tbb_rank < s_tbb_pool_size[0]
    ? s_tbb_exec[ s_tbb_pool_size[0] - ( init_tbb_rank + 1 ) ] : 0 ;

  if ( 0 == th || th->m_pool_rank != init_tbb_rank ) {
    std::ostringstream msg ;
    msg << "Kokkos::Impl::TBBExec::get_thread ERROR : "
        << "thread " << init_tbb_rank << " of " << s_tbb_pool_size[0] ;
    if ( 0 == th ) {
      msg << " does not exist" ;
    }
    else {
      msg << " has wrong thread_rank " << th->m_pool_rank ;
    }
    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  return th ;
}

//----------------------------------------------------------------------------

void TBBExec::execute_sleep( TBBExec & exec , const void * )
{
  TBBExec::global_lock();
  TBBExec::global_unlock();

  const int n = exec.m_pool_fan_size ;
  const int rank_rev = exec.m_pool_size - ( exec.m_pool_rank + 1 );

  for ( int i = 0 ; i < n ; ++i ) {
    Impl::spinwait_while_equal<int>( exec.m_pool_base[ rank_rev + (1<<i) ]->m_pool_state , TBBExec::Active );
  }

  exec.m_pool_state = TBBExec::Inactive ;
}

}
}

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

void TBBExec::verify_is_process( const std::string & name , const bool initialized )
{
  if ( ! is_process() ) {
    std::string msg( name );
    msg.append( " FAILED : Called by a worker thread, can only be called by the master process." );
    Kokkos::Impl::throw_runtime_exception( msg );
  }

  if ( initialized && 0 == s_tbb_pool_size[0] ) {
    std::string msg( name );
    msg.append( " FAILED : TBB not initialized." );
    Kokkos::Impl::throw_runtime_exception( msg );
  }
}

int TBBExec::in_parallel()
{
  // A thread function is in execution and
  // the function argument is not the special threads process argument and
  // the master process is a worker or is not the master process.
  return s_current_function &&
         ( & s_tbb_process != s_current_function_arg ) &&
         ( s_tbb_process.m_pool_base || ! is_process() );
}

// Wait for root thread to become inactive
void TBBExec::fence()
{
  if ( s_tbb_pool_size[0] ) {
    // Wait for the root thread to complete:
    Impl::spinwait_while_equal<int>( s_tbb_exec[0]->m_pool_state , TBBExec::Active );
  }

  s_current_function     = 0 ;
  s_current_function_arg = 0 ;

  // Make sure function and arguments are cleared before
  // potentially re-activating threads with a subsequent launch.
  memory_fence();
}

/** \brief  Begin execution of the asynchronous functor */
void TBBExec::start( void (*func)( TBBExec & , const void * ) , const void * arg )
{
  verify_is_process("TBBExec::start" , true );

  if ( s_current_function || s_current_function_arg ) {
    Kokkos::Impl::throw_runtime_exception( std::string( "TBBExec::start() FAILED : already executing" ) );
  }

  s_current_function     = func ;
  s_current_function_arg = arg ;

  // Make sure function and arguments are written before activating threads.
  memory_fence();

  // Activate threads:
  for ( int i = s_tbb_pool_size[0] ; 0 < i-- ; ) {
    s_tbb_exec[i]->m_pool_state = TBBExec::Active ;
  }

  if ( s_tbb_process.m_pool_size ) {
    // Master process is the root thread, run it:
    (*func)( s_tbb_process , arg );
    s_tbb_process.m_pool_state = TBBExec::Inactive ;
  }
}

//----------------------------------------------------------------------------

bool TBBExec::sleep()
{
  verify_is_process("TBBExec::sleep", true );

  if ( & execute_sleep == s_current_function ) return false ;

  fence();

  TBBExec::global_lock();

  s_current_function = & execute_sleep ;

  // Activate threads:
  for ( unsigned i = s_tbb_pool_size[0] ; 0 < i ; ) {
    s_tbb_exec[--i]->m_pool_state = TBBExec::Active ;
  }

  return true ;
}

bool TBBExec::wake()
{
  verify_is_process("TBBExec::wake", true );

  if ( & execute_sleep != s_current_function ) return false ;

  TBBExec::global_unlock();

  if ( s_tbb_process.m_pool_base ) {
    execute_sleep( s_tbb_process , 0 );
    s_tbb_process.m_pool_state = TBBExec::Inactive ;
  }

  fence();

  return true ;
}

//----------------------------------------------------------------------------

void TBBExec::execute_serial( void (*func)( TBBExec & , const void * ) )
{
  s_current_function = func ;
  s_current_function_arg = & s_tbb_process ;

  // Make sure function and arguments are written before activating threads.
  memory_fence();

  const unsigned begin = s_tbb_process.m_pool_base ? 1 : 0 ;

  for ( unsigned i = s_tbb_pool_size[0] ; begin < i ; ) {
    TBBExec & th = * s_tbb_exec[ --i ];

    th.m_pool_state = TBBExec::Active ;

    wait_yield( th.m_pool_state , TBBExec::Active );
  }

  if ( s_tbb_process.m_pool_base ) {
    s_tbb_process.m_pool_state = TBBExec::Active ;
    (*func)( s_tbb_process , 0 );
    s_tbb_process.m_pool_state = TBBExec::Inactive ;
  }

  s_current_function_arg = 0 ;
  s_current_function = 0 ;

  // Make sure function and arguments are cleared before proceeding.
  memory_fence();
}

//----------------------------------------------------------------------------

void * TBBExec::root_reduce_scratch()
{
  return s_tbb_process.reduce_memory();
}

void TBBExec::execute_resize_scratch( TBBExec & exec , const void * )
{
  typedef Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::HostSpace , void > Record ;

  if ( exec.m_scratch ) {
    Record * const r = Record::get_record( exec.m_scratch );

    exec.m_scratch = 0 ;

    Record::decrement( r );
  }

  exec.m_scratch_reduce_end = s_tbb_process.m_scratch_reduce_end ;
  exec.m_scratch_tbb_end = s_tbb_process.m_scratch_tbb_end ;

  if ( s_tbb_process.m_scratch_tbb_end ) {

    // Allocate tracked memory:
    {
      Record * const r = Record::allocate( Kokkos::HostSpace() , "thread_scratch" , s_tbb_process.m_scratch_tbb_end );

      Record::increment( r );

      exec.m_scratch = r->data();
    }

    unsigned * ptr = reinterpret_cast<unsigned *>( exec.m_scratch );

    unsigned * const end = ptr + s_tbb_process.m_scratch_tbb_end / sizeof(unsigned);

    // touch on this thread
    while ( ptr < end ) *ptr++ = 0 ;
  }
}

void * TBBExec::resize_scratch( size_t reduce_size , size_t thread_size )
{
  enum { ALIGN_MASK = Kokkos::Impl::MEMORY_ALIGNMENT - 1 };

  fence();

  const size_t old_reduce_size = s_tbb_process.m_scratch_reduce_end ;
  const size_t old_tbb_size = s_tbb_process.m_scratch_tbb_end - s_tbb_process.m_scratch_reduce_end ;

  reduce_size = ( reduce_size + ALIGN_MASK ) & ~ALIGN_MASK ;
  thread_size = ( thread_size + ALIGN_MASK ) & ~ALIGN_MASK ;

  // Increase size or deallocate completely.

  if ( ( old_reduce_size < reduce_size ) ||
       ( old_tbb_size < thread_size ) ||
       ( ( reduce_size == 0 && thread_size == 0 ) &&
         ( old_reduce_size != 0 || old_tbb_size != 0 ) ) ) {

    verify_is_process( "TBBExec::resize_scratch" , true );

    s_tbb_process.m_scratch_reduce_end = reduce_size ;
    s_tbb_process.m_scratch_tbb_end = reduce_size + thread_size ;

    execute_serial( & execute_resize_scratch );

    s_tbb_process.m_scratch = s_tbb_exec[0]->m_scratch ;
  }

  return s_tbb_process.m_scratch ;
}

//----------------------------------------------------------------------------

void TBBExec::print_configuration( std::ostream & s , const bool detail )
{
  verify_is_process("TBBExec::print_configuration",false);

  fence();

  const unsigned numa_count       = Kokkos::hwloc::get_available_numa_count();
  const unsigned cores_per_numa   = Kokkos::hwloc::get_available_cores_per_numa();
  const unsigned threads_per_core = Kokkos::hwloc::get_available_tbb_per_core();

  // Forestall compiler warnings for unused variables.
  (void) numa_count;
  (void) cores_per_numa;
  (void) threads_per_core;

  s << "Kokkos::Threads" ;

#if defined( KOKKOS_ENABLE_THREADS )
  s << " KOKKOS_ENABLE_THREADS" ;
#endif
#if defined( KOKKOS_ENABLE_HWLOC )
  s << " hwloc[" << numa_count << "x" << cores_per_numa << "x" << threads_per_core << "]" ;
#endif

  if ( s_tbb_pool_size[0] ) {
    s << " threads[" << s_tbb_pool_size[0] << "]"
      << " threads_per_numa[" << s_tbb_pool_size[1] << "]"
      << " threads_per_core[" << s_tbb_pool_size[2] << "]"
      ;
    if ( 0 == s_tbb_process.m_pool_base ) { s << " Asynchronous" ; }
    s << " ReduceScratch[" << s_current_reduce_size << "]"
      << " SharedScratch[" << s_current_shared_size << "]" ;
    s << std::endl ;

    if ( detail ) {

      for ( int i = 0 ; i < s_tbb_pool_size[0] ; ++i ) {

        TBBExec * const th = s_tbb_exec[i] ;

        if ( th ) {

          const int rank_rev = th->m_pool_size - ( th->m_pool_rank + 1 );

          s << " Thread[ " << th->m_pool_rank << " : "
            << th->m_numa_rank << "." << th->m_numa_core_rank << " ]" ;

          s << " Fan{" ;
          for ( int j = 0 ; j < th->m_pool_fan_size ; ++j ) {
            TBBExec * const thfan = th->m_pool_base[rank_rev+(1<<j)] ;
            s << " [ " << thfan->m_pool_rank << " : "
              << thfan->m_numa_rank << "." << thfan->m_numa_core_rank << " ]" ;
          }
          s << " }" ;

          if ( th == & s_tbb_process ) {
            s << " is_process" ;
          }
        }
        s << std::endl ;
      }
    }
  }
  else {
    s << " not initialized" << std::endl ;
  }
}

//----------------------------------------------------------------------------

int TBBExec::is_initialized()
{ return 0 != s_tbb_exec[0] ; }

void TBBExec::initialize( unsigned thread_count ,
                              unsigned use_numa_count ,
                              unsigned use_cores_per_numa ,
                              bool allow_asynchronous_threadpool )
{
  static const Sentinel sentinel ;

  const bool is_initialized = 0 != s_tbb_pool_size[0] ;

  unsigned thread_spawn_failed = 0 ;

  for ( int i = 0; i < TBBExec::MAX_THREAD_COUNT ; i++)
    s_tbb_exec[i] = NULL;

  if ( ! is_initialized ) {

    // If thread_count, use_numa_count, or use_cores_per_numa are zero
    // then they will be given default values based upon hwloc detection
    // and allowed asynchronous execution.

    const bool hwloc_avail = Kokkos::hwloc::available();
    const bool hwloc_can_bind = hwloc_avail && Kokkos::hwloc::can_bind_threads();

    if ( thread_count == 0 ) {
      thread_count = hwloc_avail
      ? Kokkos::hwloc::get_available_numa_count() *
        Kokkos::hwloc::get_available_cores_per_numa() *
        Kokkos::hwloc::get_available_tbb_per_core()
      : 1 ;
    }

    const unsigned thread_spawn_begin =
      hwloc::thread_mapping( "Kokkos::TBB::initialize" ,
                             allow_asynchronous_threadpool ,
                             thread_count ,
                             use_numa_count ,
                             use_cores_per_numa ,
                             s_tbb_coord );

    const std::pair<unsigned,unsigned> proc_coord = s_tbb_coord[0] ;

    if ( thread_spawn_begin ) {
      // Synchronous with s_tbb_coord[0] as the process core
      // Claim entry #0 for binding the process core.
      s_tbb_coord[0] = std::pair<unsigned,unsigned>(~0u,~0u);
    }

    s_tbb_pool_size[0] = thread_count ;
    s_tbb_pool_size[1] = s_tbb_pool_size[0] / use_numa_count ;
    s_tbb_pool_size[2] = s_tbb_pool_size[1] / use_cores_per_numa ;
    s_current_function = & execute_function_noop ; // Initialization work function

    for ( unsigned ith = thread_spawn_begin ; ith < thread_count ; ++ith ) {

      s_tbb_process.m_pool_state = TBBExec::Inactive ;

      // If hwloc available then spawned thread will
      // choose its own entry in 's_tbb_coord'
      // otherwise specify the entry.
      s_current_function_arg = (void*)static_cast<uintptr_t>( hwloc_can_bind ? ~0u : ith );

      // Make sure all outstanding memory writes are complete
      // before spawning the new thread.
      memory_fence();

      // Spawn thread executing the 'driver()' function.
      // Wait until spawned thread has attempted to initialize.
      // If spawning and initialization is successfull then
      // an entry in 's_tbb_exec' will be assigned.
      if ( TBBExec::spawn() ) {
        wait_yield( s_tbb_process.m_pool_state , TBBExec::Inactive );
      }
      if ( s_tbb_process.m_pool_state == TBBExec::Terminating ) break ;
    }

    // Wait for all spawned threads to deactivate before zeroing the function.

    for ( unsigned ith = thread_spawn_begin ; ith < thread_count ; ++ith ) {
      // Try to protect against cache coherency failure by casting to volatile.
      TBBExec * const th = ((TBBExec * volatile *)s_tbb_exec)[ith] ;
      if ( th ) {
        wait_yield( th->m_pool_state , TBBExec::Active );
      }
      else {
        ++thread_spawn_failed ;
      }
    }

    s_current_function     = 0 ;
    s_current_function_arg = 0 ;
    s_tbb_process.m_pool_state = TBBExec::Inactive ;

    memory_fence();

    if ( ! thread_spawn_failed ) {
      // Bind process to the core on which it was located before spawning occured
      if (hwloc_can_bind) {
        Kokkos::hwloc::bind_this_thread( proc_coord );
      }

      if ( thread_spawn_begin ) { // Include process in pool.
        const std::pair<unsigned,unsigned> coord = Kokkos::hwloc::get_this_tbb_coordinate();

        s_tbb_exec[0]                   = & s_tbb_process ;
        s_tbb_process.m_numa_rank       = coord.first ;
        s_tbb_process.m_numa_core_rank  = coord.second ;
        s_tbb_process.m_pool_base       = s_tbb_exec ;
        s_tbb_process.m_pool_rank       = thread_count - 1 ; // Reversed for scan-compatible reductions
        s_tbb_process.m_pool_size       = thread_count ;
        s_tbb_process.m_pool_fan_size   = fan_size( s_tbb_process.m_pool_rank , s_tbb_process.m_pool_size );
        s_tbb_pid[ s_tbb_process.m_pool_rank ] = pthread_self();
      }
      else {
        s_tbb_process.m_pool_base = 0 ;
        s_tbb_process.m_pool_rank = 0 ;
        s_tbb_process.m_pool_size = 0 ;
        s_tbb_process.m_pool_fan_size = 0 ;
      }

      // Initial allocations:
      TBBExec::resize_scratch( 1024 , 1024 );
    }
    else {
      s_tbb_pool_size[0] = 0 ;
      s_tbb_pool_size[1] = 0 ;
      s_tbb_pool_size[2] = 0 ;
    }
  }

  if ( is_initialized || thread_spawn_failed ) {

    std::ostringstream msg ;

    msg << "Kokkos::TBB::initialize ERROR" ;

    if ( is_initialized ) {
      msg << " : already initialized" ;
    }
    if ( thread_spawn_failed ) {
      msg << " : failed to spawn " << thread_spawn_failed << " threads" ;
    }

    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  // Check for over-subscription
  if( Kokkos::show_warnings() && (Impl::mpi_ranks_per_node() * long(thread_count) > Impl::processors_per_node()) ) {
    std::cout << "Kokkos::TBB::initialize WARNING: You are likely oversubscribing your CPU cores." << std::endl;
    std::cout << "                                    Detected: " << Impl::processors_per_node() << " cores per node." << std::endl;
    std::cout << "                                    Detected: " << Impl::mpi_ranks_per_node() << " MPI_ranks per node." << std::endl;
    std::cout << "                                    Requested: " << thread_count << " threads per process." << std::endl;
  }

  // Init the array for used for arbitrarily sized atomics
  Impl::init_lock_array_host_space();

  Impl::SharedAllocationRecord< void, void >::tracking_enable();

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::initialize();
  #endif
}

//----------------------------------------------------------------------------

void TBBExec::finalize()
{
  verify_is_process("TBBExec::finalize",false);

  fence();

  resize_scratch(0,0);

  const unsigned begin = s_tbb_process.m_pool_base ? 1 : 0 ;

  for ( unsigned i = s_tbb_pool_size[0] ; begin < i-- ; ) {

    if ( s_tbb_exec[i] ) {

      s_tbb_exec[i]->m_pool_state = TBBExec::Terminating ;

      wait_yield( s_tbb_process.m_pool_state , TBBExec::Inactive );

      s_tbb_process.m_pool_state = TBBExec::Inactive ;
    }

    s_tbb_pid[i] = 0 ;
  }

  if ( s_tbb_process.m_pool_base ) {
    ( & s_tbb_process )->~TBBExec();
    s_tbb_exec[0] = 0 ;
  }

  if (Kokkos::hwloc::can_bind_threads() ) {
    Kokkos::hwloc::unbind_this_thread();
  }

  s_tbb_pool_size[0] = 0 ;
  s_tbb_pool_size[1] = 0 ;
  s_tbb_pool_size[2] = 0 ;

  // Reset master thread to run solo.
  s_tbb_process.m_numa_rank       = 0 ;
  s_tbb_process.m_numa_core_rank  = 0 ;
  s_tbb_process.m_pool_base       = 0 ;
  s_tbb_process.m_pool_rank       = 0 ;
  s_tbb_process.m_pool_size       = 1 ;
  s_tbb_process.m_pool_fan_size   = 0 ;
  s_tbb_process.m_pool_state = TBBExec::Inactive ;

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::finalize();
  #endif
}

//----------------------------------------------------------------------------

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

int TBB::concurrency() {
  return thread_pool_size(0);
}

Threads & TBB::instance(int)
{
  static Threads t ;
  return t ;
}

int TBB::thread_pool_size( int depth )
{
  return Impl::s_tbb_pool_size[depth];
}

#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
int TBB::thread_pool_rank()
{
  const pthread_t pid = pthread_self();
  int i = 0;
  while ( ( i < Impl::s_tbb_pool_size[0] ) && ( pid != Impl::s_tbb_pid[i] ) ) { ++i ; }
  return i ;
}
#endif

const char* TBB::name() { return "Threads"; }
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#else
void KOKKOS_CORE_SRC_THREADS_EXEC_PREVENT_LINK_ERROR() {}
#endif /* #if defined( KOKKOS_ENABLE_THREADS ) */

