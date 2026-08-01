from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the specific content for Section 4
        title = "Application 1: Smoothing the Noise"
        lines = [
            "Convolution can smooth out noisy, jittery sensor data.",
            "A uniform kernel computes a local moving average.",
            "This creates fluid movement from raw, shaky inputs."
        ]
        self.setup_layout(title, lines)

        # Define Colors
        RED_DATA = "#FF0000"
        WHITE_WINDOW = "#FFFFFF"
        GREEN_SMOOTH = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED_DATA)
        
        # Create noisy sensor data points
        # Positioned roughly between grid rows C and D, and columns 1 to 6
        num_points = 12
        x_points = np.linspace(self.grid["C1"][0], self.grid["C6"][0], num_points)
        y_center = (self.grid["C1"][1] + self.grid["D1"][1]) / 2
        
        # Hardcoded jitters for stability
        jitters = [0.2, -0.4, 0.5, -0.1, -0.5, 0.4, 0.7, -0.3, 0.1, 0.6, -0.2, 0.3]
        noisy_pts_coords = [
            [x_points[i], y_center + jitters[i], 0] for i in range(num_points)
        ]
        
        noisy_line = VMobject(color=RED_DATA)
        noisy_line.set_points_as_corners(noisy_pts_coords)
        
        label_raw = Text("Raw Sensor Data", font_size=16, color=RED_DATA)
        # Issue 32 fix: Move from B1 to B2-B3
        self.place_in_area(label_raw, "B2", "B3")
        
        self.play(Create(noisy_line), Write(label_raw), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE_WINDOW)
        
        # Create the kernel window (Box)
        # It covers roughly 3 points
        box_width = (x_points[1] - x_points[0]) * 2.2
        box_height = 1.2
        window_box = Rectangle(
            width=box_width, 
            height=box_height, 
            color=WHITE_WINDOW, 
            fill_color=WHITE_WINDOW, 
            fill_opacity=0.2,
            stroke_width=2
        )
        # Start the window at the first triplet center (index 1)
        window_box.move_to(noisy_pts_coords[1])
        
        label_kernel = Text("Moving Window [1/3, 1/3, 1/3]", font_size=16, color=WHITE_WINDOW)
        # Issue 33 fix: Move from B5 to B4-B6, scale 0.8
        self.place_in_area(label_kernel, "B4", "B6", scale_factor=0.8)

        self.play(FadeIn(window_box), FadeIn(label_kernel))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN_SMOOTH)
        
        # Calculate smoothed points (moving average)
        smoothed_pts_coords = []
        for i in range(1, num_points - 1):
            avg_y = (jitters[i-1] + jitters[i] + jitters[i+1]) / 3.0
            smoothed_pts_coords.append([x_points[i], y_center + avg_y, 0])
            
        label_smooth = Text("Filtered Output", font_size=16, color=GREEN_SMOOTH)
        # Issue 34 fix: Move from E1 to E2
        self.place_at_grid(label_smooth, "E2")
        self.add(label_smooth)

        # Animating the sliding window and the appearing smooth line
        smooth_segments = VGroup()
        
        for i in range(len(smoothed_pts_coords) - 1):
            # Move window to the next center point
            target_pos = noisy_pts_coords[i+2]
            
            # Line segment between current smoothed point and next
            segment = Line(
                smoothed_pts_coords[i], 
                smoothed_pts_coords[i+1], 
                color=GREEN_SMOOTH, 
                stroke_width=4
            )
            
            self.play(
                window_box.animate.move_to(target_pos),
                Create(segment),
                run_time=0.6,
                rate_func=linear
            )
            smooth_segments.add(segment)

        self.wait(2)
