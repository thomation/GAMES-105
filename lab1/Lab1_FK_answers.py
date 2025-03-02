import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        stack = []
        parent_index = -1
        for line in lines:
            if 'ROOT' in line or 'JOINT' in line:
                name = line.split()[1]
                joint_name.append(name)
                joint_parent.append(parent_index)
                parent_index = len(joint_name) - 1
                stack.append(parent_index)
            elif 'End Site' in line:
                joint_name.append(f"{joint_name[parent_index]}_end")
                joint_parent.append(parent_index)
                parent_index = len(joint_name) - 1
                stack.append(parent_index)
            elif 'OFFSET' in line:
                offset = np.array([float(x) for x in line.split()[1:]])
                joint_offset.append(offset)
            elif '}' in line:
                stack.pop()
                if stack:
                    parent_index = stack[-1]

    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))
    joint_orientations[:, 3] = 1  # Initialize w component of quaternion to 1
    # print(f"joint_name size {len(joint_name)}")
    # print(f"motion data size {len(motion_data[frame_id])}")
    motion_start_index = 3 
    for i in range(len(joint_name)):
        # print(f"handle joint {i} {joint_name[i]} parent is {joint_parent[i]}")
        if joint_parent[i] == -1:
            joint_positions[i] = motion_data[frame_id, :3]
            joint_orientations[i] = R.from_euler('XYZ', motion_data[frame_id, motion_start_index:motion_start_index + 3], degrees=True).as_quat()
            motion_start_index += 3
        elif(not joint_name[i].endswith("_end")):
            parent_pos = joint_positions[joint_parent[i]]
            parent_ori = R.from_quat(joint_orientations[joint_parent[i]])
            local_pos = joint_offset[i]
            local_ori = R.from_euler('XYZ', motion_data[frame_id, motion_start_index:motion_start_index + 3], degrees=True)
            joint_positions[i] = parent_pos + parent_ori.apply(local_pos)
            joint_orientations[i] = (parent_ori * local_ori).as_quat()
            motion_start_index += 3
    return joint_positions, joint_orientations

def compute_motion_data_index(joint_names, joint_name):
    index = 3
    for i in range(len(joint_names)):
        if joint_names[i] == joint_name:
            return index 
        elif(not joint_names[i].endswith("_end")):
            index += 3
    # raise exception
    raise ValueError(f"Joint name {joint_name} not found in joint names list")


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)

    motion_data_A = load_motion_data(A_pose_bvh_path)
    motion_data_T = np.zeros_like(motion_data_A)
    
    joint_map = {name: i for i, name in enumerate(joint_name_A)}
    
    for frame_id in range(motion_data_A.shape[0]):
        joint_positions_A, joint_orientations_A = part2_forward_kinematics(joint_name_A, joint_parent_A, joint_offset_A, motion_data_A, frame_id)
        
        for i, name in enumerate(joint_name_T):
            if name in joint_map:
                idx_A = joint_map[name]
                if joint_parent_T[i] == -1:
                    motion_data_T[frame_id, :6] = motion_data_A[frame_id, :6]
                elif(not joint_name_T[i].endswith("_end")):
                    to_start_index = compute_motion_data_index(joint_name_T, joint_name_T[i])
                    from_start_index = compute_motion_data_index(joint_name_A, joint_name_A[idx_A])
                    motion_data_T[frame_id, to_start_index: 3 + to_start_index] = motion_data_A[frame_id, from_start_index : 3 + from_start_index]
    
    return motion_data_T
