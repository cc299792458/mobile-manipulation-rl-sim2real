import numpy as np
from scipy.optimize import fsolve

import time

r_target = 300  # 指定 r 的目标值
z_target = 100  # 指定 z 的目标值
beta_target = 0

def degree2int32(angle):
    return [int(item * 4096 / 360) for item in angle]

def rad2int32(angle):
    return [int(item * 4096 / (2*np.pi)) for item in angle]

# 修改方程组定义，包含额外的参数
def equations(vars, r_target, z_target, beta_target):
    theta1, theta2, theta3 = vars    
    a, b1, b2, c1, d1, d2 = 80.25, 117.5+25, 20.75, 137.5, 138, 20.5
    r = b1 * np.sin(theta1) + b2 * np.cos(theta1) + c1 * np.cos(theta1 + theta2) + d1 * np.cos(theta1 + theta2 + theta3) - d2 * np.sin(theta1 + theta2 + theta3)  
    z = a + b1 * np.cos(theta1) - b2 * np.sin(theta1) - c1 * np.sin(theta1 + theta2) - d1 * np.sin(theta1 + theta2 + theta3) - d2 * np.cos(theta1 + theta2 + theta3) 
    
    beta = theta1 + theta2 + theta3
    return [r - r_target, z - z_target, beta - beta_target]

# 在主函数或调用代码中指定目标值
r_target = 300
z_target = 100
beta_target = 0

# 初始猜测值
theta1_guess, theta2_guess, theta3_guess = 0, 0, 0
start_time = time.time()
solution = fsolve(equations, (theta1_guess, theta2_guess, theta3_guess), args=(r_target, z_target, beta_target))
end_time = time.time()
print("Solve time:", end_time - start_time)
print('The solution is:', rad2int32(solution))
print('The value of the equations at this solution is:', equations(solution, r_target, z_target, beta_target))
