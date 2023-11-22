/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu>

/* SkePU user functions */

/*
float userfunction(...)
{
	// your code here
}

// more user functions...

*/
float add(float a, float b){
	return a + b;
}

float mult(float a, float b){
	return a * b;
}

int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
//	spec.setCPUThreads(<integer value>);
	skepu::setGlobalBackendSpec(spec);
	
	
	/* Skeleton instances */
	auto dotprod = skepu::MapReduce <2 >( mult , add );
	auto add2 = skepu::Reduce(add);
	auto mult2 = skepu::Map<2>(mult);
//	auto instance = skepu::Map(userfunction);
// ...
	
	/* SkePU containers */
	skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f), v3(size, 0.0f);

	v1.randomize(0, 10);
	v2.randomize(0, 10);

	//std :: cout << "v1: " << v1 << "\n";
	//std :: cout << "v2: " << v2 << "\n";

	
	
	/* Compute and measure time */
	float resComb, resSep;
	
	auto timeComb = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		resComb = dotprod(v1, v2);
	});
	
	auto timeSep = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		mult2(v3, v1, v2);
		resSep = add2(v3);
	});
	
	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";
	
	
	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";
	
	return 0;
}

