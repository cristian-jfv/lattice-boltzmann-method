#include "ulbm.hpp"
#include "utils.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/diagflat.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>


#define icf0 inter_coll_f.index({torch::indexing::Ellipsis,0})
#define icf1 inter_coll_f.index({torch::indexing::Ellipsis,1})
#define icf2 inter_coll_f.index({torch::indexing::Ellipsis,2})
#define icf3 inter_coll_f.index({torch::indexing::Ellipsis,3})
#define icf4 inter_coll_f.index({torch::indexing::Ellipsis,4})
#define icf5 inter_coll_f.index({torch::indexing::Ellipsis,5})
#define icf6 inter_coll_f.index({torch::indexing::Ellipsis,6})
#define icf7 inter_coll_f.index({torch::indexing::Ellipsis,7})
#define icf8 inter_coll_f.index({torch::indexing::Ellipsis,8})

#define cT0 cT.index({torch::indexing::Ellipsis,0})
#define cT1 cT.index({torch::indexing::Ellipsis,1})
#define cT2 cT.index({torch::indexing::Ellipsis,2})
#define cT3 cT.index({torch::indexing::Ellipsis,3})
#define cT4 cT.index({torch::indexing::Ellipsis,4})
#define cT5 cT.index({torch::indexing::Ellipsis,5})
#define cT6 cT.index({torch::indexing::Ellipsis,6})
#define cT7 cT.index({torch::indexing::Ellipsis,7})
#define cT8 cT.index({torch::indexing::Ellipsis,8})

#define ux m1.index({torch::indexing::Ellipsis,0})
#define uy m1.index({torch::indexing::Ellipsis,1})

ulbm::d2q9::kbc::kbc(int R, int C, double s2):
//m1{torch::zeros({R,C,2}, torch::kCUDA)},
s2{s2},
is2{1.0/s2}
//inter_coll_f{torch::zeros({R,C,9}, torch::kCUDA)},
//cT{torch::zeros({R,C,9}, torch::kCUDA)}
{
  using torch::indexing::Ellipsis;
  const int Q = 9;

  iequi_f = torch::zeros({R,C,Q}, torch::kCUDA);
  coll_f = torch::zeros({R,C,Q}, torch::kCUDA);
  adve_f = torch::zeros({R,C,Q}, torch::kCUDA);

  S = torch::ones({R,C,Q}, torch::kCUDA);
  S.index({Ellipsis,3}) = s2;
  S.index({Ellipsis,4}) = s2;
  S.index({Ellipsis,5}) = s2;

  gamma = torch::zeros({R,C}, torch::kCUDA);
  delta_s = torch::zeros({R,C,Q}, torch::kCUDA);
  delta_h = torch::zeros({R,C,Q}, torch::kCUDA);
  inter_coll_f = torch::zeros({R,C,Q}, torch::kCUDA);
  // icf0 = inter_coll_f.index({Ellipsis,0});
  // icf1 = inter_coll_f.index({Ellipsis,1});
  // icf2 = inter_coll_f.index({Ellipsis,2});
  // icf3 = inter_coll_f.index({Ellipsis,3});
  // icf4 = inter_coll_f.index({Ellipsis,4});
  // icf5 = inter_coll_f.index({Ellipsis,5});
  // icf6 = inter_coll_f.index({Ellipsis,6});
  // icf7 = inter_coll_f.index({Ellipsis,7});
  // icf8 = inter_coll_f.index({Ellipsis,8});

  cT = torch::zeros({R,C,Q}, torch::kCUDA);
  // cT0 = cT.index({torch::indexing::Ellipsis,0});
  // cT1 = cT.index({torch::indexing::Ellipsis,1});
  // cT2 = cT.index({torch::indexing::Ellipsis,2});
  // cT3 = cT.index({torch::indexing::Ellipsis,3});
  // cT4 = cT.index({torch::indexing::Ellipsis,4});
  // cT5 = cT.index({torch::indexing::Ellipsis,5});
  // cT6 = cT.index({torch::indexing::Ellipsis,6});
  // cT7 = cT.index({torch::indexing::Ellipsis,7});
  // cT8 = cT.index({torch::indexing::Ellipsis,8});
  Ts = torch::zeros({R,C,Q}, torch::kCUDA);
  Th = torch::zeros({R,C,Q}, torch::kCUDA);

  m0 = torch::zeros({R,C}, torch::kCUDA);
  m1 = torch::zeros({R,C,2}, torch::kCUDA);
  // ux = m1.index({torch::indexing::Ellipsis,0});
  // uy = m1.index({torch::indexing::Ellipsis,1});
  ux2 = torch::zeros({R,C}, torch::kCUDA);
  uy2 = torch::zeros({R,C}, torch::kCUDA);
  cmx = torch::zeros({R,C,Q}, torch::kCUDA);
  cmy = torch::zeros({R,C,Q}, torch::kCUDA);
  cmx2 = torch::zeros({R,C,Q}, torch::kCUDA);
  cmy2 = torch::zeros({R,C,Q}, torch::kCUDA);
  cm_buffer = torch::zeros({R,C}, torch::kCUDA);
}

void ulbm::d2q9::kbc::collide()
{
  using torch::indexing::Ellipsis;
  eval_central_momenta();
  eval_s_matrix();

  // 1.- Inplace substraction |cT⟩ - |cTeq⟩ -> |cT⟩
  cT.index({Ellipsis,0}).add_(-m0);
  cT.index({Ellipsis,3}).add_(-2.0*cs2*m0);
  cT.index({Ellipsis,8}).add_(-cs4*m0);
  // 2.- Inplace elementwise multiplication S|cT⟩ -> |cT⟩
  cT.mul_(S);
  // 3.- invN|cT⟩ -> inter_coll_f
  inter_coll_f.index_put_({Ellipsis,0}, cT0 );
  inter_coll_f.index_put_({Ellipsis,1}, cT0*ux + cT1 );
  inter_coll_f.index_put_({Ellipsis,2}, cT0*uy + cT2 );
  inter_coll_f.index_put_({Ellipsis,3}, cT0*(ux2+uy2) + 2.0*cT1*ux + 2.0*cT2*uy + cT3 );
  inter_coll_f.index_put_({Ellipsis,4}, cT0*(ux2-uy2) + 2.0*cT1*ux - 2.0*cT2*uy + cT4 );
  inter_coll_f.index_put_({Ellipsis,5}, cT0*ux*uy + cT1*uy + cT2*ux + cT5 );
  inter_coll_f.index_put_({Ellipsis,6}, cT0*ux2*uy + 2.0*cT1*ux*uy + cT2*ux2 + 0.5*cT3*uy + 0.5*cT4*uy + 2.0*cT5*ux + cT6 );
  inter_coll_f.index_put_({Ellipsis,7}, cT0*ux*uy2 + cT1*uy2 + 2.0*cT2*ux*uy + 0.5*cT3*ux - 0.5*cT4*ux + 2.0*cT5*uy + cT7 );
  inter_coll_f.index_put_({Ellipsis,8}, cT0*ux2*uy2 + 2.0*cT1*ux*uy2 + 2.0*cT2*ux2*uy + 0.5*cT3*(ux2 + uy2) - 0.5*cT4*(ux2 - uy2) + 4.0*cT5*ux*uy + 2.0*cT6*uy + 2.0*cT7*ux + cT8 );
  // 4.- minusinvM*inter_coll_f -> coll_f
  coll_f.index_put_({Ellipsis,0}, icf0 - icf3 + icf8);
  coll_f.index_put_({Ellipsis,1}, 0.5*icf1 + 0.25*icf3 + 0.25*icf4 - 0.5*icf7 - 0.5*icf8);
  coll_f.index_put_({Ellipsis,2}, 0.5*icf2 + 0.25*icf3 - 0.25*icf4 - 0.5*icf6 - 0.5*icf8);
  coll_f.index_put_({Ellipsis,3}, -0.5*icf1 + 0.25*icf3 + 0.25*icf4 + 0.5*icf7 - 0.5*icf8);
  coll_f.index_put_({Ellipsis,4}, -0.5*icf2 + 0.25*icf3 - 0.25*icf4 + 0.5*icf6 - 0.5*icf8);
  coll_f.index_put_({Ellipsis,5}, 0.25*(icf5 + icf6 + icf7 + icf8));
  coll_f.index_put_({Ellipsis,6}, 0.25*(-icf5 + icf6 - icf7 + icf8));
  coll_f.index_put_({Ellipsis,7}, 0.25*(icf5 - icf6 - icf7 + icf8));
  coll_f.index_put_({Ellipsis,8}, 0.25*(-icf5 - icf6 + icf7 + icf8));
  coll_f.multiply_(-1.0);
  // 5.- Inplace addition coll_f + adve_f -> coll_f
  coll_f.add_(adve_f);
}

void ulbm::d2q9::kbc::eval_s_matrix()
{
  using torch::indexing::Ellipsis;

  eval_gamma();
  S.index_put_({Ellipsis,6},gamma*s2);
  S.index_put_({Ellipsis,7},gamma*s2);
  S.index_put_({Ellipsis,8},gamma*s2);
}

void ulbm::d2q9::kbc::eval_gamma()
{
  eval_m1_components();
  eval_delta_s();
  eval_delta_h();
  eval_iequilibrium();
  gamma.copy_(
    is2 - (1.0 - is2)*torch::sum(delta_s*delta_h*iequi_f,-1)
    /torch::sum(delta_h*delta_h*iequi_f,-1)
  );
}

void ulbm::d2q9::kbc::eval_m1_components()
{
  using torch::indexing::Ellipsis;
  ux2.copy_(ux.pow(2));
  uy2.copy_(uy.pow(2));
}

void ulbm::d2q9::kbc::eval_delta_s()
{
  using torch::indexing::Ellipsis;
  const utils::indices C3 = {Ellipsis,3};
  const utils::indices C4 = {Ellipsis,4};
  const utils::indices C5 = {Ellipsis,5};

  // Assumes central momenta are already calculated
  delta_s.index_put_({Ellipsis, 0},
    -0.5*cT.index(C4)*(ux2 - uy2) + 4.0*cT.index(C5)*ux*uy - cs4*m0 - m0*(ux2*uy2 - ux2 - uy2 + 1) + (cT.index(C3) - 2.0*cs2*m0)*(0.5*ux2+0.5*uy2-1.0)
  );
  delta_s.index_put_({Ellipsis, 1},
    0.25*cT.index(C4)*(ux2-uy2+ux+1) - cT.index(C5)*uy*(2.0*ux+1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - ux2 + uy2*ux - ux) - 0.25*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2+ux-1.0)
  );
  delta_s.index_put_({Ellipsis, 2},
    -0.25*cT.index(C4)*(-ux2+uy2+uy+1) - cT.index(C5)*ux*(2.0*uy+1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - uy2 + ux2*uy - uy) - 0.25*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2+uy-1.0)
  );
  delta_s.index_put_({Ellipsis, 3},
    0.25*cT.index(C4)*(ux2-uy2-ux+1) - cT.index(C5)*uy*(2.0*ux-1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - ux2 - uy2*ux + ux) - 0.25*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2-ux-1.0)
  );
  delta_s.index_put_({Ellipsis, 4},
    0.25*cT.index(C4)*(ux2-uy2+uy-1) - cT.index(C5)*ux*(2.0*uy-1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - uy2 - ux2*uy + uy) - 0.25*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2-uy-1.0)
  );
  delta_s.index_put_({Ellipsis, 5},
    -0.125*cT.index(C4)*(ux2-uy2+ux-uy) + cT.index(C5)*(ux*uy+0.5*ux+0.5*uy+0.25) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2+ux2*uy+uy2*ux+ux*uy) + 0.125*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2+ux+uy)
  );
  delta_s.index_put_({Ellipsis, 6},
    0.125*cT.index(C4)*(-ux2+uy2+ux+uy) + cT.index(C5)*(ux*uy+0.5*ux-0.5*uy-0.25) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2+ux2*uy-uy2*ux-ux*uy) + 0.125*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2-ux+uy)
  );
  delta_s.index_put_({Ellipsis, 7},
    -0.125*cT.index(C4)*(ux2-uy2-ux+uy) + cT.index(C5)*(ux*uy-0.5*ux-0.5*uy+0.25) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2-ux2*uy-uy2*ux+ux*uy) + 0.125*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2-ux-uy)
  );
  delta_s.index_put_({Ellipsis, 8},
    -0.125*cT.index(C4)*(ux2-uy2+ux+uy) + cT.index(C5)*(ux*uy-0.5*ux+0.5*uy-0.25) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2-ux2*uy+uy2*ux-ux*uy) + 0.125*(cT.index(C3)-2.0*cs2*m0)*(ux2+uy2+ux-uy)
  );
}

void ulbm::d2q9::kbc::eval_delta_h()
{
  using torch::indexing::Ellipsis;
  const utils::indices C6 = {Ellipsis,6};
  const utils::indices C7 = {Ellipsis,7};
  const utils::indices C8 = {Ellipsis,8};
  // Assumes central momenta are already calculated
  delta_h.index_put_({Ellipsis,0},
     2.0*cT.index(C6)*uy + 2.0*cT.index(C7)*ux + cT.index(C8) - 2.0*cs2*m0*(0.5*ux2+0.5*uy2 - 1.0) - cs4*m0 - m0*(ux2*uy2-ux2-uy2+1.0)
  );
  delta_h.index_put_({Ellipsis,1},
    -cT.index(C6)*uy - cT.index(C7)*(ux+0.5) - 0.5*cT.index(C8) + 0.5*cs2*m0*(ux2+uy2+ux-1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - ux2 + uy2*ux - ux)
  );
  delta_h.index_put_({Ellipsis,2},
    -cT.index(C6)*(uy+0.5) - cT.index(C7)*ux - 0.5*cT.index(C8) + 0.5*cs2*m0*(ux2+uy2+uy-1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 + ux2*uy - uy2 - uy)
  );
  delta_h.index_put_({Ellipsis,3},
    -cT.index(C6)*uy - cT.index(C7)*(ux-0.5) - 0.5*cT.index(C8) + 0.5*cs2*m0*(ux2+uy2-ux-1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - ux2 - uy2*ux + ux)
  );
  delta_h.index_put_({Ellipsis,4},
    -cT.index(C6)*(uy-0.5) - cT.index(C7)*ux - 0.5*cT.index(C8) + 0.5*cs2*m0*(ux2+uy2-uy-1.0) + 0.5*cs4*m0 + 0.5*m0*(ux2*uy2 - ux2*uy - uy2 + uy)
  );
  delta_h.index_put_({Ellipsis,5},
    cT.index(C6)*(0.5*uy+0.25) + cT.index(C7)*(0.5*ux+0.25) + 0.25*cT.index(C8) - 0.25*cs2*m0*(ux2+uy2+ux+uy) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2 + ux2+uy + uy2*ux +ux*uy)
  );
  delta_h.index_put_({Ellipsis,6},
    cT.index(C6)*(0.5*uy+0.25) + cT.index(C7)*(0.5*ux-0.25) + 0.25*cT.index(C8) - 0.25*cs2*m0*(ux2+uy2-ux+uy) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2 + ux2+uy - uy2*ux -ux*uy)
  );
  delta_h.index_put_({Ellipsis,7},
    cT.index(C6)*(0.5*uy-0.25) + cT.index(C7)*(0.5*ux-0.25) + 0.25*cT.index(C8) - 0.25*cs2*m0*(ux2+uy2-ux-uy) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2 - ux2+uy - uy2*ux +ux*uy)
  );
  delta_h.index_put_({Ellipsis,8},
    cT.index(C6)*(0.5*uy-0.25) + cT.index(C7)*(0.5*ux+0.25) + 0.25*cT.index(C8) - 0.25*cs2*m0*(ux2+uy2+ux-uy) - 0.25*cs4*m0 - 0.25*m0*(ux2*uy2 - ux2+uy + uy2*ux -ux*uy)
  );
}

void ulbm::d2q9::kbc::eval_iequilibrium()
{
  using torch::indexing::Ellipsis;

  iequi_f.index_put_({Ellipsis,0}, 2.0*cs2*(0.5*ux2 + 0.5*uy2 - 1.0) + cs4 + ux2*uy2 - ux2 - uy2 + 1.0 );
  iequi_f.index_put_({Ellipsis,1}, 0.5*( -cs2*(ux2 + uy2 + ux - 1.0) - cs4 - ux2*uy2 + ux2 - uy2*ux + ux ) );
  iequi_f.index_put_({Ellipsis,2}, 0.5*( -cs2*(ux2 + uy2 + uy - 1.0) - cs4 - ux2*uy2 - ux2*uy + uy2 + uy ) );
  iequi_f.index_put_({Ellipsis,3}, 0.5*( -cs2*(ux2 + uy2 - ux - 1.0) - cs4 - ux2*uy2 + ux2 + uy2*ux - ux ) );
  iequi_f.index_put_({Ellipsis,4}, 0.5*( -cs2*(ux2 + uy2 - uy - 1.0) - cs4 - ux2*uy2 + ux2*uy + uy2 - uy ) );
  iequi_f.index_put_({Ellipsis,5}, 0.25*( cs2*(ux2 + uy2 + ux + uy) + cs4 + ux2*uy2 + ux2*uy + uy2*ux + ux*uy ) );
  iequi_f.index_put_({Ellipsis,6}, 0.25*( cs2*(ux2 + uy2 - ux + uy) + cs4 + ux2*uy2 + ux2*uy - uy2*ux - ux*uy ) );
  iequi_f.index_put_({Ellipsis,7}, 0.25*( cs2*(ux2 + uy2 - ux - uy) + cs4 + ux2*uy2 - ux2*uy - uy2*ux + ux*uy ) );
  iequi_f.index_put_({Ellipsis,8}, 0.25*( cs2*(ux2 + uy2 + ux - uy) + cs4 + ux2*uy2 - ux2*uy + uy2*ux - ux*uy ) );

  iequi_f.mul_(m0.unsqueeze(-1));
  iequi_f.pow_(-1);
}

void ulbm::d2q9::kbc::eval_equilibrium(torch::Tensor& equi_f)
{
  using torch::indexing::Ellipsis;

  equi_f.index_put_({Ellipsis,0}, 2.0*cs2*(0.5*ux2 + 0.5*uy2 - 1.0) + cs4 + ux2*uy2 - ux2 - uy2 + 1.0 );
  equi_f.index_put_({Ellipsis,1}, 0.5*( -cs2*(ux2 + uy2 + ux - 1.0) - cs4 - ux2*uy2 + ux2 - uy2*ux + ux ) );
  equi_f.index_put_({Ellipsis,2}, 0.5*( -cs2*(ux2 + uy2 + uy - 1.0) - cs4 - ux2*uy2 - ux2*uy + uy2 + uy ) );
  equi_f.index_put_({Ellipsis,3}, 0.5*( -cs2*(ux2 + uy2 - ux - 1.0) - cs4 - ux2*uy2 + ux2 + uy2*ux - ux ) );
  equi_f.index_put_({Ellipsis,4}, 0.5*( -cs2*(ux2 + uy2 - uy - 1.0) - cs4 - ux2*uy2 + ux2*uy + uy2 - uy ) );
  equi_f.index_put_({Ellipsis,5}, 0.25*( cs2*(ux2 + uy2 + ux + uy) + cs4 + ux2*uy2 + ux2*uy + uy2*ux + ux*uy ) );
  equi_f.index_put_({Ellipsis,6}, 0.25*( cs2*(ux2 + uy2 - ux + uy) + cs4 + ux2*uy2 + ux2*uy - uy2*ux - ux*uy ) );
  equi_f.index_put_({Ellipsis,7}, 0.25*( cs2*(ux2 + uy2 - ux - uy) + cs4 + ux2*uy2 - ux2*uy - uy2*ux + ux*uy ) );
  equi_f.index_put_({Ellipsis,8}, 0.25*( cs2*(ux2 + uy2 + ux - uy) + cs4 + ux2*uy2 - ux2*uy + uy2*ux - ux*uy ) );

  equi_f.mul_(m0.unsqueeze(-1));
}

void ulbm::d2q9::kbc::eval_central_momenta()
{
  using torch::indexing::Ellipsis;
  using torch::sum;

  // cmx
  cm_buffer.copy_(-m1.index({Ellipsis,0}));
  cmx.index_put_({Ellipsis,0}, cm_buffer);
  cmx.index_put_({Ellipsis,2}, cm_buffer);
  cmx.index_put_({Ellipsis,4}, cm_buffer);
  cm_buffer.copy_(1.0-m1.index({Ellipsis,0}));
  cmx.index_put_({Ellipsis,1}, cm_buffer);
  cmx.index_put_({Ellipsis,5}, cm_buffer);
  cmx.index_put_({Ellipsis,8}, cm_buffer);
  cm_buffer.copy_(-1.0-m1.index({Ellipsis,0}));
  cmx.index_put_({Ellipsis,3}, cm_buffer);
  cmx.index_put_({Ellipsis,6}, cm_buffer);
  cmx.index_put_({Ellipsis,7}, cm_buffer);
  // cmy
  cm_buffer.copy_(-m1.index({Ellipsis,1}));
  cmy.index_put_({Ellipsis,0}, cm_buffer);
  cmy.index_put_({Ellipsis,1}, cm_buffer);
  cmy.index_put_({Ellipsis,3}, cm_buffer);
  cm_buffer.copy_(1.0-m1.index({Ellipsis,1}));
  cmy.index_put_({Ellipsis,2}, cm_buffer);
  cmy.index_put_({Ellipsis,5}, cm_buffer);
  cmy.index_put_({Ellipsis,6}, cm_buffer);
  cm_buffer.copy_(-1.0-m1.index({Ellipsis,1}));
  cmy.index_put_({Ellipsis,4}, cm_buffer);
  cmy.index_put_({Ellipsis,7}, cm_buffer);
  cmy.index_put_({Ellipsis,8}, cm_buffer);
  // cmx2
  cmx2.copy_(cmx.pow(2));
  // cmy2
  cmy2.copy_(cmy.pow(2));

  // T
  // k00
  cT.index_put_({Ellipsis,0}, adve_f.sum(-1));
  // k10
  cT.index_put_({Ellipsis,1}, sum(adve_f*cmx,-1));
  // k01
  cT.index_put_({Ellipsis,2}, sum(adve_f*cmy,-1));
  // k20 + k02
  cT.index_put_({Ellipsis,3}, sum(adve_f*(cmx2 + cmy2),-1));
  // k20 - k02
  cT.index_put_({Ellipsis,4}, sum(adve_f*(cmx2 - cmy2),-1));
  // k11
  cT.index_put_({Ellipsis,5}, sum(adve_f*cmx*cmy,-1));
  // k21
  cT.index_put_({Ellipsis,6}, sum(adve_f*cmx2*cmy,-1));
  // k12
  cT.index_put_({Ellipsis,7}, sum(adve_f*cmx*cmy2,-1));
  // k22
  cT.index_put_({Ellipsis,8}, sum(adve_f*cmx2*cmy2,-1));
}

void ulbm::d2q9::kbc::advect()
{
  using torch::indexing::Ellipsis;
  using torch::indexing::Slice;
  using torch::indexing::None;
  // Advect
  // f0
  adve_f.index({Ellipsis}) = coll_f.index({Ellipsis}).clone().detach();

  // f1
  //print("f1");
  adve_f.index({Slice(1,None), Slice(), 1}) = coll_f.index({Slice(0,-1), Slice(), 1}).clone().detach();
  adve_f.index({0, Slice(), 1}) = coll_f.index({-1, Slice(), 1}).clone().detach();

  // f2
  //print("f2");
  adve_f.index({Slice(), Slice(1,None), 2}) = coll_f.index({Slice(), Slice(0,-1), 2}).clone().detach();
  adve_f.index({Slice(), 0, 2}) = coll_f.index({Slice(), -1, 2}).clone().detach();

  // f3
  //print("f3");
  adve_f.index({Slice(0,-1), Slice(), 3}) = coll_f.index({Slice(1,None), Slice(), 3}).clone().detach();
  adve_f.index({-1, Slice(), 3}) = coll_f.index({0,Slice(),3}).clone().detach();

  // f4
  //print("f4");
  adve_f.index({Slice(), Slice(0,-1), 4}) = coll_f.index({Slice(), Slice(1,None), 4}).clone().detach();
  adve_f.index({Slice(), -1, 4}) = coll_f.index({Slice(), 0, 4}).clone().detach();

  // f5
  //print("f5");
  adve_f.index({Slice(1,None), Slice(1,None), 5}) = coll_f.index({Slice(0,-1), Slice(0,-1),5}).clone().detach();
  adve_f.index({0, Slice(1,None), 5}) = coll_f.index({-1, Slice(0,-1), 5}).clone().detach();
  adve_f.index({Slice(1,None), 0, 5}) = coll_f.index({Slice(0,-1), -1, 5}).clone().detach();
  adve_f.index({0, 0, 5}) = coll_f.index({-1, -1, 5}).clone().detach();

  // f6
  //print("f6");
  adve_f.index({Slice(0,-1), Slice(1,None), 6}) = coll_f.index({Slice(1,None), Slice(0,-1), 6}).clone().detach();
  adve_f.index({-1, Slice(1,None), 6}) = coll_f.index({0, Slice(0,-1), 6}).clone().detach();
  adve_f.index({Slice(0,-1), 0, 6}) = coll_f.index({Slice(1,None), -1, 6}).clone().detach();
  adve_f.index({-1, 0, 6}) = coll_f.index({0, -1, 6}).clone().detach();

  // f7
  //print("f7");
  adve_f.index({Slice(0,-1), Slice(0,-1), 7}) = coll_f.index({Slice(1,None), Slice(1,None), 7}).clone().detach();
  adve_f.index({-1, Slice(0,-1), 7}) = coll_f.index({0, Slice(1,None), 7}).clone().detach();
  adve_f.index({Slice(0,-1), -1, 7}) = coll_f.index({Slice(1,None), 0, 7}).clone().detach();
  adve_f.index({-1, -1, 7}) = coll_f.index({0, 0, 7}).clone().detach();

  // f8
  //print("f8");
  adve_f.index({Slice(1,None), Slice(0,-1), 8}) = coll_f.index({Slice(0,-1), Slice(1,None), 8}).clone().detach();
  adve_f.index({0, Slice(0,-1), 8}) = coll_f.index({-1, Slice(1,None), 8});
  adve_f.index({Slice(1,None), -1, 8}) = coll_f.index({Slice(0,-1), 0, 8});
  adve_f.index({0, -1, 8}) = coll_f.index({-1, 0, 8}).clone().detach();

}

