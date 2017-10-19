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

#ifndef KOKKOS_TBBEXEC_HPP
#define KOKKOS_TBBEXEC_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TBB )

#if !defined(_TBB)
#error "You enabled Kokkos TBB support without enabling TBB in the compiler!"
#endif

#include <Kokkos_TBB.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include <Kokkos_Atomic.hpp>

#include <Kokkos_UniqueToken.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

#include <omp.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

class TBBExec;

extern int g_openmp_hardware_max_threads;

extern __thread int t_openmp_hardware_id;
extern __thread TBBExec * t_openmp_instance;

//----------------------------------------------------------------------------
/** \brief  Data for TBB thread execution */

class TBBExec {
public:

  friend class Kokkos::TBB ;

  enum { MAX_THREAD_COUNT = 512 };

  void clear_thread_data();

  static void validate_partition( const int nthreads
                                , int & num_partitions
                                , int & partition_size
                                );

private:
  TBBExec( int arg_pool_size )
    : m_pool_size{ arg_pool_size }
    , m_level{ omp_get_level() }
    , m_pool()
  {}

  ~TBBExec()
  {
    clear_thread_data();
  }

  int m_pool_size;
  int m_level;

  HostThreadTeamData * m_pool[ MAX_THREAD_COUNT ];

public:

  static void verify_is_master( const char * const );

  void resize_thread_data( size_t pool_reduce_bytes
                         , size_t team_reduce_bytes
                         , size_t team_shared_bytes
                         , size_t thread_local_bytes );

  inline
  HostThreadTeamData * get_thread_data() const noexcept
  { return m_pool[ m_level == omp_get_level() ? 0 : omp_get_thread_num() ]; }

  inline
  HostThreadTeamData * get_thread_data( int i ) const noexcept
  { return m_pool[i]; }
};

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

inline TBB::TBB() noexcept
{}

inline
bool TBB::is_initialized() noexcept
{ return Impl::t_openmp_instance != nullptr; }

inline
bool TBB::in_parallel( TBB const& ) noexcept
{
  //t_openmp_instance is only non-null on a master thread
  return   !Impl::t_openmp_instance
         || Impl::t_openmp_instance->m_level < omp_get_level()
         ;
}

inline
int TBB::thread_pool_size() noexcept
{
  return   TBB::in_parallel()
         ? omp_get_num_threads()
         : Impl::t_openmp_instance->m_pool_size
         ;
}

KOKKOS_INLINE_FUNCTION
int TBB::thread_pool_rank() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_openmp_instance ? 0 : omp_get_thread_num();
#else
  return -1 ;
#endif
}

inline
void TBB::fence( TBB const& instance ) noexcept {}

inline
bool TBB::is_asynchronous( TBB const& instance ) noexcept
{ return false; }

template <typename F>
void TBB::partition_master( F const& f
                             , int num_partitions
                             , int partition_size
                             )
{
  if (omp_get_nested()) {
    using Exec = Impl::TBBExec;

    Exec * prev_instance = Impl::t_openmp_instance;

    Exec::validate_partition( prev_instance->m_pool_size, num_partitions, partition_size );

    TBB::memory_space space;

    #pragma omp parallel num_threads(num_partitions)
    {
      void * const ptr = space.allocate( sizeof(Exec) );

      Impl::t_openmp_instance = new (ptr) Exec( partition_size );

      size_t pool_reduce_bytes  =   32 * partition_size ;
      size_t team_reduce_bytes  =   32 * partition_size ;
      size_t team_shared_bytes  = 1024 * partition_size ;
      size_t thread_local_bytes = 1024 ;

      Impl::t_openmp_instance->resize_thread_data( pool_reduce_bytes
                                                 , team_reduce_bytes
                                                 , team_shared_bytes
                                                 , thread_local_bytes
                                                 );

      omp_set_num_threads(partition_size);
      f( omp_get_thread_num(), omp_get_num_threads() );

      Impl::t_openmp_instance->~Exec();
      space.deallocate( Impl::t_openmp_instance, sizeof(Exec) );
      Impl::t_openmp_instance = nullptr;
    }

    Impl::t_openmp_instance  = prev_instance;
  }
  else {
    // nested openmp not enabled
    f(0,1);
  }
}


namespace Experimental {

template<>
class MasterLock<TBB>
{
public:
  void lock()     { omp_set_lock( &m_lock );   }
  void unlock()   { omp_unset_lock( &m_lock ); }
  bool try_lock() { return static_cast<bool>(omp_test_lock( &m_lock )); }

  MasterLock()  { omp_init_lock( &m_lock ); }
  ~MasterLock() { omp_destroy_lock( &m_lock ); }

  MasterLock( MasterLock const& ) = delete;
  MasterLock( MasterLock && )     = delete;
  MasterLock & operator=( MasterLock const& ) = delete;
  MasterLock & operator=( MasterLock && )     = delete;

private:
  omp_lock_t m_lock;

};

template<>
class UniqueToken< TBB, UniqueTokenScope::Instance>
{
public:
  using execution_space = TBB;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken( execution_space const& = execution_space() ) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Kokkos::TBB::thread_pool_size();
      #else
      return 0 ;
      #endif
    }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const  noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Kokkos::TBB::thread_pool_rank();
      #else
      return 0 ;
      #endif
    }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release( int ) const noexcept {}
};

template<>
class UniqueToken< TBB, UniqueTokenScope::Global>
{
public:
  using execution_space = TBB;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken( execution_space const& = execution_space() ) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Kokkos::Impl::g_openmp_hardware_max_threads ;
      #else
      return 0 ;
      #endif
    }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Kokkos::Impl::t_openmp_hardware_id ;
      #else
      return 0 ;
      #endif
    }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release( int ) const noexcept {}
};

} // namespace Experimental


#if !defined( KOKKOS_DISABLE_DEPRECATED )

inline
int TBB::thread_pool_size( int depth )
{
  return depth < 2
         ? thread_pool_size()
         : 1;
}

KOKKOS_INLINE_FUNCTION
int TBB::hardware_thread_id() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_openmp_hardware_id;
#else
  return -1 ;
#endif
}

inline
int TBB::max_hardware_threads() noexcept
{
  return Impl::g_openmp_hardware_max_threads;
}

#endif // KOKKOS_DISABLE_DEPRECATED

} // namespace Kokkos

#endif
#endif /* #ifndef KOKKOS_TBBEXEC_HPP */

