#include <Kokkos_Core.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <Kokkos_Random.hpp>

  template <typename ViewT>
  KOKKOS_INLINE_FUNCTION constexpr auto Determinant(ViewT F)
      -> std::enable_if_t<Kokkos::is_view<ViewT>::value && ViewT::rank == 2,
                          double>
  {
      return (F(0, 0) * F(1, 1) * F(2, 2) + F(0, 1) * F(1, 2) * F(2, 0) +
              F(0, 2) * F(1, 0) * F(2, 1) -
              (F(0, 2) * F(1, 1) * F(2, 0) + F(0, 1) * F(1, 0) * F(2, 2) +
               F(0, 0) * F(1, 2) * F(2, 1)));
  }

template <typename ExeSpace, typename ViewT>
void GenerateTestData(ViewT data) {
  using memory_space = typename ExeSpace::memory_space;
  // finite difference should return dPK2dU. So, we can analyze two cases.
  Kokkos::Random_XorShift64_Pool<memory_space> random(13718);
  Kokkos::fill_random(data, random, 1.0);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExeSpace>(0, data.extent(0)), KOKKOS_LAMBDA(int i) {
        auto data_i = Kokkos::subview(data, i, Kokkos::ALL(), Kokkos::ALL());
        while (Determinant(data_i) < 0.5)
        {
          data_i(0, 0) += 1.0;
          data_i(1, 1) += 1.0;
          data_i(2, 2) += 1.0;
        }
      });
}

template <typename ExeSpace, typename ViewT, int N=3>
void BatchedSVD(ViewT matrices) {
    using memory_space = typename ExeSpace::memory_space;
    Kokkos::View<double * [N][N], memory_space> Us("Us", matrices.extent(0));
    Kokkos::View<double * [N], memory_space> Ss("Ss", matrices.extent(0));
    Kokkos::View<double * [N][N], memory_space> Vts("Vts", matrices.extent(0));
    Kokkos::View<double * [N], memory_space> works("works", matrices.extent(0));
    Kokkos::View<double * [N][N], memory_space> matrices_copy("matrices_copy",
                                                matrices.extent(0));
    // make a copy of the input data to avoid overwriting it
    Kokkos::deep_copy(matrices_copy, matrices);
    auto policy = Kokkos::RangePolicy<ExeSpace>(0, matrices.extent(0));
    Kokkos::parallel_for(
        "polar decomposition", policy, KOKKOS_LAMBDA(int i) {
          auto matrix_copy =
              Kokkos::subview(matrices_copy, i, Kokkos::ALL(), Kokkos::ALL());
          auto U = Kokkos::subview(Us, i, Kokkos::ALL(), Kokkos::ALL());
          auto S = Kokkos::subview(Ss, i, Kokkos::ALL());
          auto Vt = Kokkos::subview(Vts, i, Kokkos::ALL(), Kokkos::ALL());
          auto work = Kokkos::subview(works, i, Kokkos::ALL());
          KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag{},
                                           matrix_copy, U, S, Vt, work);
    });
    auto Us_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Us);
    auto Ss_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},Ss);
    auto Vts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},Vts);
    auto matrices_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},matrices);
    for(int i=0; i<Us.extent(0); ++i ) {
      std::cout<<"Input Matrix\n";
      for(int j=0; j<matrices_h.extent(1); ++j) {
        for(int k=0; k<matrices_h.extent(2); ++k) {
          std::cout<<matrices_h(i,j,k) <<" ";

        }
        std::cout<<"\n";
      }
      std::cout<<"--------------------\n";
      std::cout<<"U\n";
      for(int j=0; j<Us_h.extent(1); ++j) {
        for(int k=0; k<Us_h.extent(2); ++k) {
          std::cout<<Us_h(i,j,k) <<" ";

        }
        std::cout<<"\n";
      }
      std::cout<<"--------------------\n";
      std::cout<<"Vt\n";
      for(int j=0; j<Vts_h.extent(1); ++j) {
        for(int k=0; k<Vts_h.extent(2); ++k) {
          std::cout<<Vts_h(i,j,k) <<" ";
        }
        std::cout<<"\n";
      }
      std::cout<<"--------------------\n";
      std::cout<<"S\n";
      for(int j=0; j<Ss_h.extent(1); ++j) {
        std::cout<<Ss_h(i,j)<<" ";
      }
      std::cout<<"\n";
      std::cout<<"====================\n";
    }
}

int main() {
  Kokkos::ScopeGuard sg{};
  constexpr int num_tests = 4;
  Kokkos::View<double * [3][3], Kokkos::CudaSpace> data_cuda("data", num_tests);
  Kokkos::View<double * [3][3], Kokkos::HostSpace> data_host("data", num_tests);

  std::cout<<"CUDA\n";
  GenerateTestData<Kokkos::Cuda>(data_cuda);
  BatchedSVD<Kokkos::Cuda>(data_cuda);

  std::cout<<"SERIAL\n";
  GenerateTestData<Kokkos::Serial>(data_host);
  BatchedSVD<Kokkos::Serial>(data_host);

}
