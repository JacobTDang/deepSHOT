"""
Interactive Interpolation Demo

This shows your interpolation idea in action and compares different approaches.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def linear_interpolation(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    """
    Linear interpolation between two points
    
    Args:
        p1: Starting point (x1, y1)
        p2: Ending point (x2, y2)  
        t: Time factor (0.0 = p1, 1.0 = p2, 0.5 = middle)
    
    Returns:
        Interpolated point
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Your "average" idea when t=0.5
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)

def demonstrate_interpolation():
    """Show different interpolation strategies"""
    
    # Test data - frames from our earlier example
    frame_1 = (102, 105)  # Known
    frame_4 = (108, 115)  # Known
    # Frames 2 and 3 are missing - let's interpolate!
    
    print("=== Your Interpolation Idea in Action ===\n")
    
    # Method 1: Your "average" approach (linear interpolation)
    print("1. Linear Interpolation (Your Idea):")
    frame_2_linear = linear_interpolation(frame_1, frame_4, 1/3)  # 1/3 of way from 1 to 4
    frame_3_linear = linear_interpolation(frame_1, frame_4, 2/3)  # 2/3 of way from 1 to 4
    
    print(f"   Frame 1: {frame_1} (known)")
    print(f"   Frame 2: {frame_2_linear} (interpolated)")
    print(f"   Frame 3: {frame_3_linear} (interpolated)")
    print(f"   Frame 4: {frame_4} (known)")
    
    # Method 2: What if motion isn't linear? (quadratic interpolation)
    print("\n2. Quadratic Interpolation (curved motion):")
    # Assume some acceleration - like a badminton stroke speeding up
    def quadratic_interpolation(p1, p2, t):
        # Non-linear interpolation with acceleration
        # t^2 makes motion speed up over time
        t_curved = t * t  # Acceleration curve
        return linear_interpolation(p1, p2, t_curved)
    
    frame_2_quad = quadratic_interpolation(frame_1, frame_4, 1/3)
    frame_3_quad = quadratic_interpolation(frame_1, frame_4, 2/3)
    
    print(f"   Frame 1: {frame_1} (known)")
    print(f"   Frame 2: {frame_2_quad} (accelerating)")
    print(f"   Frame 3: {frame_3_quad} (accelerating)")
    print(f"   Frame 4: {frame_4} (known)")
    
    # Method 3: Velocity-based prediction (like Kalman filter concept)
    print("\n3. Velocity-Based Prediction:")
    # Calculate velocity from frame 1 to frame 4
    dt = 3  # 3 frame gap
    velocity_x = (frame_4[0] - frame_1[0]) / dt
    velocity_y = (frame_4[1] - frame_1[1]) / dt
    
    print(f"   Calculated velocity: ({velocity_x:.1f}, {velocity_y:.1f}) per frame")
    
    frame_2_velocity = (frame_1[0] + 1*velocity_x, frame_1[1] + 1*velocity_y)
    frame_3_velocity = (frame_1[0] + 2*velocity_x, frame_1[1] + 2*velocity_y)
    
    print(f"   Frame 1: {frame_1} (known)")
    print(f"   Frame 2: {frame_2_velocity} (velocity prediction)")
    print(f"   Frame 3: {frame_3_velocity} (velocity prediction)")
    print(f"   Frame 4: {frame_4} (known)")
    
    # Create visualization
    create_interpolation_plot(frame_1, frame_4, frame_2_linear, frame_3_linear,
                             frame_2_quad, frame_3_quad, frame_2_velocity, frame_3_velocity)

def create_interpolation_plot(frame_1, frame_4, f2_lin, f3_lin, f2_quad, f3_quad, f2_vel, f3_vel):
    """Visualize different interpolation methods"""
    
    plt.figure(figsize=(12, 8))
    
    # Known points
    plt.scatter(*frame_1, color='green', s=100, label='Frame 1 (known)', marker='o')
    plt.scatter(*frame_4, color='green', s=100, label='Frame 4 (known)', marker='o')
    
    # Linear interpolation (your idea)
    plt.scatter(*f2_lin, color='blue', s=80, label='Linear interpolation', marker='^')
    plt.scatter(*f3_lin, color='blue', s=80, marker='^')
    plt.plot([frame_1[0], f2_lin[0], f3_lin[0], frame_4[0]], 
             [frame_1[1], f2_lin[1], f3_lin[1], frame_4[1]], 
             'b--', alpha=0.7, label='Linear path')
    
    # Quadratic interpolation
    plt.scatter(*f2_quad, color='red', s=80, label='Quadratic interpolation', marker='s')
    plt.scatter(*f3_quad, color='red', s=80, marker='s')
    
    # Velocity-based
    plt.scatter(*f2_vel, color='orange', s=80, label='Velocity-based', marker='d')
    plt.scatter(*f3_vel, color='orange', s=80, marker='d')
    
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.title('Different Interpolation Methods for Missing Frames')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add frame labels
    plt.annotate('Frame 1', frame_1, xytext=(5, 5), textcoords='offset points')
    plt.annotate('Frame 4', frame_4, xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('data/processed/interpolation_comparison.png', dpi=150)
    print(f"\n📊 Visualization saved to: data/processed/interpolation_comparison.png")
    plt.show()

def badminton_stroke_analysis():
    """Apply this to badminton stroke scenario"""
    print("\n=== Badminton Stroke Application ===")
    print("Imagine a forehand stroke where tracking is lost mid-swing:")
    
    # Realistic badminton stroke coordinates
    backswing = (200, 300)      # Racket back, ready position
    contact = (350, 180)        # Contact point (racket meets shuttle)
    
    print(f"Backswing position: {backswing}")
    print(f"Contact position: {contact}")
    print(f"Missing frames: 2-3 frames during the swing")
    
    # Your interpolation approach
    mid_swing = linear_interpolation(backswing, contact, 0.5)
    print(f"Your interpolated mid-swing: {mid_swing}")
    
    # Compare to simple "copy previous frame" approach
    print(f"Simple copying approach: {backswing} (just repeats)")
    
    print(f"\n🎯 Key insight: Your method predicts the racket is moving toward contact!")
    print(f"    This is much better for badminton analysis than staying static.")

if __name__ == "__main__":
    demonstrate_interpolation()
    badminton_stroke_analysis()