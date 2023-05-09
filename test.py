
k_base = [1]
global ans
ans = 0
def dfs(L, R, i, tmp):
    if i == len(k_base):
        if tmp>=L and tmp<=R:
            global ans
            print(":",tmp)
            ans += 1
        return 
    if tmp > R:
        return
	# 当前这一位1
    dfs(L, R, i+1, tmp+k_base[i])
	# 当前这一位0
    dfs(L, R, i+1, tmp)
def count_k_power_nums(L, R, K):
    k = K
    while k<R:
        k_base.append(k)
        k*=K
    # print(k_base)
    dfs(L, R, 0, 0)
    return ans

# 测试
print(count_k_power_nums(100, 11000000000, 10000000))  # 输出 1