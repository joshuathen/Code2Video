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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize Scene Layout
        title_text = "Prerequisites: Energy and Momentum"
        lecture_lines = [
            "Momentum is conserved in every collision.",
            "Energy conservation follows a quadratic relationship.",
            "We plot velocities as a point in state space."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Defining Colors
        MOMENTUM_COLOR = "#00FF00"
        ENERGY_COLOR = "#FFFF00"
        AXES_COLOR = "#FFFFFF"
        POINT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(MOMENTUM_COLOR))
        
        # Display formula 'mv + MV = p' in green (#00FF00) at 'B4'.
        momentum_formula = MathTex("mv + MV = p", color=MOMENTUM_COLOR)
        self.place_at_grid(momentum_formula, "B4", scale_factor=0.9)
        self.play(Write(momentum_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(ENERGY_COLOR))
        
        # Display formula '1/2mv² + 1/2MV² = E' in yellow (#FFFF00) at 'C4'.
        energy_formula = MathTex(r"\frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E", color=ENERGY_COLOR)
        self.place_at_grid(energy_formula, "C4", scale_factor=0.9)
        self.play(Write(energy_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Clear formulas to make room for the graph
        self.play(FadeOut(momentum_formula), FadeOut(energy_formula))
        
        # Issue 27: Poor grid space utilization. Fix: Graph in area A2-F6.
        # Draw a white coordinate plane (#FFFFFF) centered at 'E4' (per storyboard) 
        # but the Critic wants it in A2-F6. We prioritize Critic.
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": AXES_COLOR, "include_tip": True},
        )
        v_label = MathTex("v", color=AXES_COLOR).scale(0.8)
        V_label = MathTex("V", color=AXES_COLOR).scale(0.8)
        
        # Container for the graph
        graph_elements = VGroup(axes)
        self.place_in_area(graph_elements, 'A2', 'F6', scale_factor=0.8)
        
        # Position labels relative to the axes
        v_label.next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        V_label.next_to(axes.y_axis.get_top(), UP, buff=0.1)
        graph_elements.add(v_label, V_label)

        # Issue 29: Incorrect trajectory positioning. Fix: ellipse_trajectory at 'B3' to 'E5'.
        ellipse_trajectory = Ellipse(width=3, height=2, color=ENERGY_COLOR)
        self.place_in_area(ellipse_trajectory, 'B3', 'E5', scale_factor=0.9)
        
        # Issue 28: Missing visual anchor: velocity_point at 'C4'.
        velocity_point = Dot(color=POINT_COLOR)
        self.place_at_grid(velocity_point, 'C4', scale_factor=0.6)
        
        # Animation sequence
        self.play(Create(axes), Write(v_label), Write(V_label))
        self.play(Create(ellipse_trajectory))
        self.play(FadeIn(velocity_point))
        self.wait(2)
