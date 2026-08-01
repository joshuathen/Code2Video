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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: The Laws of the Game",
            [
                "Energy and momentum are conserved in elastic collisions.",
                "We map block velocities to a 2D state space.",
                "Every state is a point on a coordinate system."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Display formulas for Momentum and Energy conservation in #FFFFFF.
        self.lecture[0].set_color(WHITE)
        
        momentum_formula = MathTex(r"m_1v_1 + m_2v_2 = \text{const}", color=WHITE)
        energy_formula = MathTex(r"\frac{1}{2}m_1v_1^2 + \frac{1}{2}m_2v_2^2 = \text{const}", color=WHITE)
        formulas = VGroup(momentum_formula, energy_formula).arrange(DOWN, buff=0.5)
        
        # Issue 23: Fix positioning of formulas
        self.place_in_area(formulas, "A3", "B6", scale_factor=0.8)
        self.play(Write(formulas))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw axes #888888 for velocities v1 and v2.
        self.lecture[1].set_color("#888888")
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": "#888888", "include_tip": True},
            tips=True
        )
        labels = axes.get_axis_labels(
            x_label=MathTex("v_1", color="#888888"), 
            y_label=MathTex("v_2", color="#888888")
        )
        axes_group = VGroup(axes, labels)
        
        # Issue 24: Fix positioning of axes
        self.place_in_area(axes_group, "C3", "F6", scale_factor=0.8)
        
        self.play(Create(axes_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Place state dot #FF00FF that jumps per collision.
        self.lecture[2].set_color("#FF00FF")
        
        # Define some arbitrary points for the "jumps"
        # v1 and v2 relative to the axes
        points = [
            axes.c2p(1, 1),
            axes.c2p(-1.5, 0.5),
            axes.c2p(0.5, -1.2),
            axes.c2p(-0.8, -0.8)
        ]
        
        state_dot = Dot(point=points[0], color="#FF00FF", radius=0.1)
        dot_label = MathTex("(v_1, v_2)", color="#FF00FF", font_size=24)
        
        # Using updater for persistent connection
        dot_label.add_updater(lambda d: d.next_to(state_dot, UR, buff=0.1))
        
        self.play(FadeIn(state_dot), FadeIn(dot_label))
        self.wait(0.5)
        
        # Jumping animation
        for pt in points[1:]:
            self.play(state_dot.animate.move_to(pt), run_time=0.8)
            self.wait(0.3)
            
        self.wait(2)
