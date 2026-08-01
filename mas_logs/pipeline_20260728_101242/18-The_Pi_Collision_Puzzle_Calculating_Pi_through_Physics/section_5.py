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

class Section5Scene(TeachingScene):
    def construct(self):
        # Colors from storyboard and visual requirements
        RED = "#FF4D4F"
        YELLOW = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"
        
        lecture_lines = [
            "Each collision acts like a reflection within the circle.",
            "Wall hits and block hits create distinct bounce patterns.",
            "The process resembles a beam of light bouncing inside."
        ]
        
        self.setup_layout("The Geometric Insight: The Bouncing Beam", lecture_lines)
        
        # Set colors for lecture lines to match animation elements
        self.lecture[0].set_color(YELLOW)
        self.lecture[1].set_color(WHITE_COLOR)
        self.lecture[2].set_color(RED)

        # === Animation for Lecture Line 1 ===
        # Place a yellow circle to represent the state space
        # Issue 32: Use larger grid area A1 to F6
        circle = Circle(radius=2.4, color=YELLOW)
        self.place_in_area(circle, 'A1', 'F6')
        circle_center = circle.get_center()
        self.play(Create(circle))
        
        # Place a red point on the boundary to represent the initial state
        initial_theta = PI/6
        initial_point_pos = circle.point_at_angle(initial_theta)
        initial_dot = Dot(initial_point_pos, color=RED)
        self.play(FadeIn(initial_dot))
        
        # A white line reflects across a diameter to represent a block-block collision
        # Diameter at angle alpha
        alpha = PI/3
        d1 = circle_center + circle.radius * np.array([np.cos(alpha), np.sin(alpha), 0])
        d2 = circle_center - circle.radius * np.array([np.cos(alpha), np.sin(alpha), 0])
        diameter = Line(d1, d2, color=WHITE_COLOR, stroke_opacity=0.3)
        self.play(Create(diameter))
        
        # Calculate reflected point: theta' = 2*alpha - theta
        reflected_theta_1 = 2*alpha - initial_theta
        reflected_pos_1 = circle.point_at_angle(reflected_theta_1)
        reflection_path = Line(initial_point_pos, reflected_pos_1, color=WHITE_COLOR)
        
        self.play(Create(reflection_path))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A white line reflects vertically to represent a wall collision
        # Reflection across horizontal (u-axis): theta' = -theta
        reflected_theta_2 = -reflected_theta_1
        reflected_pos_2 = circle.point_at_angle(reflected_theta_2)
        wall_reflection_path = Line(reflected_pos_1, reflected_pos_2, color=WHITE_COLOR)
        
        self.play(Create(wall_reflection_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Clean up previous reflection lines to prepare for the zigzag beam
        self.play(
            FadeOut(diameter),
            FadeOut(reflection_path),
            FadeOut(wall_reflection_path),
            FadeOut(initial_dot)
        )
        
        # A zigzag beam [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/beam.svg]
        # Load the beam asset (Issue 24)
        beam_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beam.svg")
        beam_icon.set_color(WHITE_COLOR)
        beam_icon.scale(0.15)
        
        current_angle = initial_theta
        d_theta = 0.73 # irrational-ish angle for non-repeating pattern
        
        beam_points = [circle.point_at_angle(current_angle)]
        for _ in range(8):
            current_angle = (current_angle + d_theta) % (2*PI)
            beam_points.append(circle.point_at_angle(current_angle))
            
        # Place initial beam icon
        beam_icon.move_to(beam_points[0])
        self.add(beam_icon)
            
        # Draw the beam segments and flashes sequentially
        for i in range(len(beam_points)-1):
            start_p = beam_points[i]
            end_p = beam_points[i+1]
            segment = Line(start_p, end_p, color=WHITE_COLOR)
            
            # Animation: segment draws and beam icon follows
            self.play(
                Create(segment),
                beam_icon.animate.move_to(end_p),
                run_time=0.4
            )
            
            # Each point where the beam hits the circle boundary flashes in red (#FF4D4F)
            hit_point = Dot(end_p, color=RED, radius=0.06)
            self.play(Flash(hit_point, color=RED, flash_radius=0.25, num_lines=8), run_time=0.2)
            self.add(hit_point)
            
        self.wait(2)
