#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>

#include "RandomGenerator.hpp"

namespace tpcc
{
using u64 = std::uint64_t;

class ZipfianGenerator
{
  private:
   u64 n;
   double theta;
   double alpha;
   double zetan;
   double eta;

   static double zeta(u64 n, double theta)
   {
      double ans = 0.0;
      for (u64 i = 1; i <= n; i++)
         ans += std::pow(1.0 / static_cast<double>(i), theta);
      return ans;
   }

  public:
   ZipfianGenerator(u64 domainSize, double theta) : n(domainSize ? (domainSize - 1) : 0), theta(theta)
   {
      if (n < 1) {
         alpha = 0.0;
         zetan = 1.0;
         eta = 0.0;
         return;
      }
      alpha = 1.0 / (1.0 - theta);
      zetan = zeta(n, theta);
      eta = (1.0 - std::pow(2.0 / static_cast<double>(n), 1.0 - theta)) /
            (1.0 - zeta(2, theta) / zetan);
   }

   u64 rand() const
   {
      if (n == 0)
         return 0;

      constexpr double constant = 1'000'000'000'000'000'000.0;
      u64 i = RandomGenerator::getRandU64(0, 1'000'000'000'000'000'001ULL);
      double u = static_cast<double>(i) / constant;
      double uz = u * zetan;
      if (uz < 1.0)
         return 1;
      if (uz < (1.0 + std::pow(0.5, theta)))
         return 2;
      double inner = eta * u - eta + 1.0;
      inner = std::max(inner, 0.0);
      u64 ret = 1 + static_cast<u64>(n * std::pow(inner, alpha));
      if (ret > n)
         ret = n;
      return ret;
   }
};

inline u64 fnvHash(u64 value)
{
   constexpr u64 offset = 0xCBF29CE484222325ULL;
   constexpr u64 prime = 1099511628211ULL;
   u64 hash = offset;
   for (int i = 0; i < 8; i++) {
      u64 octet = value & 0xFFULL;
      value >>= 8;
      hash ^= octet;
      hash *= prime;
   }
   return hash;
}

class ScrambledZipfGenerator
{
  public:
   u64 min;
   u64 max;
   u64 n;
   double theta;
   ZipfianGenerator zipf;

   ScrambledZipfGenerator(u64 min, u64 max, double theta)
       : min(min), max(max), n((max > min) ? (max - min) : 1), theta(theta), zipf((max > min) ? ((max - min) * 2) : 1, theta) {}

   u64 rand()
   {
      u64 zipfValue = zipf.rand();
      return min + (fnvHash(zipfValue) % n);
   }
};

}  // namespace tpcc

