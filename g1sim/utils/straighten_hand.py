from curobo.config_io import load_yaml, write_yaml

data = load_yaml('config/unitree_g1_custom.yml')
csp = data['kinematics']['cspace']
names = csp['joint_names']
defaults = csp['default_joint_position']

straight_arm_joints = {
	'left_shoulder_pitch_joint': 0.00,
	'right_shoulder_pitch_joint': 0.00,
	'left_shoulder_roll_joint': 0.00,
	'right_shoulder_roll_joint': 0.00,
	'left_shoulder_yaw_joint': 0.00,
	'right_shoulder_yaw_joint': 0.00,
	'left_elbow_joint': 1.39,
	'right_elbow_joint': 1.39,
	'left_wrist_roll_joint': 0.00,
	'right_wrist_roll_joint': 0.00,
	'left_wrist_pitch_joint': 0.00,
	'right_wrist_pitch_joint': 0.00,
	'left_wrist_yaw_joint': 0.00,
	'right_wrist_yaw_joint': 0.00,
}

for joint, val in straight_arm_joints.items():
	if joint in names:
		defaults[names.index(joint)] = val

write_yaml(data, 'config/unitree_g1_custom.yml')
print('Updated to straight, relaxed vertical arms in config/unitree_g1_custom.yml')
