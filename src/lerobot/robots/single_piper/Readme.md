# 一、数据回放修改点
1. 修改使能
在 connect 函数中，将
```py
# self._enable_single()
```
注释解开

2. 修改下面的代码
```py
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    # if not self.is_connected:
    #     raise DeviceNotConnectedError(f"{self} is not connected.")
    # goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
    # goal_list = [goal_pos[key].item() for key in goal_pos]
    # print("goal_pos",goal_list)
    # if self._enable_flag:
    #     self.control_arm(goal_list,60)
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")
    if self.motors is None:
        raise DeviceNotConnectedError(f"self motor value is None.")
    
    action_dict = self.motors.copy()
    return action_dict
```
将返回motors数据的代码改为控制代码








