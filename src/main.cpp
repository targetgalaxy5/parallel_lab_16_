#include <algorithm>
#include <execution>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <cmath>
#include <string>
#include <iomanip>

using steady_clock = std::chrono::steady_clock;
using ms = std::chrono::duration<double, std::milli>;

std::vector<int> make_random_sequence(std::size_t n, uint32_t seed, int mn = -1000000, int mx = 1000000) {
    std::mt19937 r(seed);
    std::uniform_int_distribution<int> d(mn, mx);
    std::vector<int> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) v.push_back(d(r));
    return v;
}

long long lib_adj(const std::vector<int>& a) {
    if (a.size() < 2) return 0;
    std::vector<long long> d(a.size());
    std::adjacent_difference(a.begin(), a.end(), d.begin());
    long long m = 0;
    for (size_t i = 1; i < d.size(); i++) {
        long long v = std::llabs(d[i]);
        if (v > m) m = v;
    }
    return m;
}

template <class Exec>
long long pol_adj(const std::vector<int>& a, Exec p) {
    if (a.size() < 2) return 0;
    std::size_t n = a.size() - 1;
    std::vector<long long> d(n);
    std::transform(p, a.begin(), a.begin() + n, d.begin(),
                   [](int x, int y){ return std::llabs((long long)y - (long long)x); });
    return std::reduce(p, d.begin(), d.end(), 0LL, [](auto x, auto y){ return x > y ? x : y; });
}

long long custom_adj(const std::vector<int>& a, unsigned K) {
    if (a.size() < 2) return 0;
    std::size_t N = a.size();
    if (K < 1) K = 1;
    if (K > N - 1) K = N - 1;
    std::size_t total = N - 1;
    std::size_t base = total / K;
    std::size_t rem = total % K;
    std::size_t ps = 0;
    std::vector<std::future<long long>> f;
    f.reserve(K);
    auto w = [&a](size_t s, size_t e){
        if (s >= e) return 0LL;
        long long m = 0;
        for (size_t i = s + 1; i <= e; i++) {
            long long v = std::llabs((long long)a[i] - (long long)a[i - 1]);
            if (v > m) m = v;
        }
        return m;
    };
    for (unsigned k = 0; k < K; k++) {
        size_t cnt = base + (k < rem ? 1 : 0);
        if (!cnt) {
            f.emplace_back(std::async(std::launch::deferred, [](){ return 0LL; }));
            continue;
        }
        size_t pe = ps + cnt - 1;
        size_t s = ps;
        size_t e = pe + 1;
        f.emplace_back(std::async(std::launch::async, w, s, e));
        ps += cnt;
    }
    long long g = 0;
    for (auto& x : f) {
        long long v = x.get();
        if (v > g) g = v;
    }
    return g;
}

struct Row {
    std::string alg, pol;
    size_t n;
    unsigned K, hw;
    double t;
    long long mx;
    uint32_t seed;
};

int main() {
    std::vector<size_t> S = {1000, 10000, 100000, 1000000};
    std::vector<unsigned> KV = {1,2,4,8,16,32};
    unsigned T = 5;
    uint32_t base = 1111;
    unsigned hw = std::thread::hardware_concurrency();
    if (!hw) hw = 2;
    std::vector<Row> R;
    for (auto n : S) for (unsigned ti = 0; ti < T; ti++) {
        uint32_t seed = base + ti;
        auto a = make_random_sequence(n, seed);
        {
            double s = 0; long long v = 0;
            for (int r = 0; r < 3; r++) {
                auto t0 = steady_clock::now();
                v = lib_adj(a);
                auto t1 = steady_clock::now();
                s += std::chrono::duration_cast<ms>(t1 - t0).count();
            }
            R.push_back({"library_adjacent_difference","none",n,0,hw,s/3.0,v,seed});
        }
        {
            double s = 0; long long v = 0;
            for (int r = 0; r < 3; r++) {
                auto t0 = steady_clock::now();
                v = pol_adj(a, std::execution::seq);
                auto t1 = steady_clock::now();
                s += std::chrono::duration_cast<ms>(t1 - t0).count();
            }
            R.push_back({"transform+reduce","seq",n,0,hw,s/3.0,v,seed});
        }
        {
            double s = 0; long long v = 0;
            for (int r = 0; r < 3; r++) {
                auto t0 = steady_clock::now();
                v = pol_adj(a, std::execution::par);
                auto t1 = steady_clock::now();
                s += std::chrono::duration_cast<ms>(t1 - t0).count();
            }
            R.push_back({"transform+reduce","par",n,0,hw,s/3.0,v,seed});
        }
        {
            double s = 0; long long v = 0;
            for (int r = 0; r < 3; r++) {
                auto t0 = steady_clock::now();
                v = pol_adj(a, std::execution::par_unseq);
                auto t1 = steady_clock::now();
                s += std::chrono::duration_cast<ms>(t1 - t0).count();
            }
            R.push_back({"transform+reduce","par_unseq",n,0,hw,s/3.0,v,seed});
        }
        for (unsigned K : KV) {
            if (K > n) break;
            double s = 0; long long v = 0;
            for (int r = 0; r < 3; r++) {
                auto t0 = steady_clock::now();
                v = custom_adj(a,K);
                auto t1 = steady_clock::now();
                s += std::chrono::duration_cast<ms>(t1 - t0).count();
            }
            R.push_back({"custom_split","custom",n,K,hw,s/3.0,v,seed});
        }
    }
    std::ofstream out("results.csv");
    out<<"algorithm,policy,n,K,hw_threads,avg_time_ms,max_value,seed\n";
    for (auto& x : R)
        out<<x.alg<<","<<x.pol<<","<<x.n<<","<<x.K<<","<<x.hw<<","<<std::fixed<<std::setprecision(6)<<x.t<<","<<x.mx<<","<<x.seed<<"\n";
    out.close();
    std::cout<<"Finished. Results saved to results.csv\n";
    return 0;
}
