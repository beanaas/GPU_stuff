**Write an explanation on how CAS can be used to implement protection for concurrent use of data structures**
CAS is a hardware implementation of an atomic operation that can be used to access a shared resource. This means we can implement a lock free multithreaded algorithm. 
With CAS we can guarantee that one operation will always succeed, therefore it is lock free.  And if upper bound on #retrys is guaranteed, then #wait-free. 

```c++
int counter; 
do{
    old_val = counter; 
    new_val = old_val + 1;
    CAS(&counter, old_val, &new_val); 
}while(new_val != old_val) //proceed until CAS suceeds. 
```

**Sketch a scenario featuring several threads raising the ABA problem**
