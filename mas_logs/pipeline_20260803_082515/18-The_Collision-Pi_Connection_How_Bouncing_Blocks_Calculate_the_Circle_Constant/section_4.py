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
        # Title and lecture lines
        title = "Mapping to Geometry: The Velocity Space"
        lines = [
            "Let's plot both velocities on a two-dimensional grid.",
            "Energy conservation forces the velocities onto an ellipse.",
            "Each collision moves our state to a new point."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color(YELLOW)
        
        # Create axes v (#00FF00) and V (#FFA500)
        # Using Axes mobject
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "stroke_width": 2},
            x_axis_config={"color": "#00FF00"},
            y_axis_config={"color": "#FFA500"}
        )
        
        # Labels for axes
        v_label = MathTex("v", color="#00FF00", font_size=24)
        V_label = MathTex("V", color="#FFA500", font_size=24)
        
        # Position labels relative to axes
        v_label.next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        V_label.next_to(axes.y_axis.get_end(), UP, buff=0.1)
        
        axes_group = VGroup(axes, v_label, V_label)
        # Fix: Move from 'B1'-'F6' to 'B2'-'F5' with scale 0.9 as per issue 31
        self.place_in_area(axes_group, 'B2', 'F5', scale_factor=0.9)
        
        self.play(Create(axes), FadeIn(v_label), FadeIn(V_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture line colors
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # White ellipse for conservation constraint
        # Energy: 1/2 m v^2 + 1/2 M V^2 = E
        ellipse = axes.plot_parametric_curve(
            lambda t: np.array([2 * np.cos(t), 1.2 * np.sin(t), 0]),
            t_range=[0, TAU],
            color=WHITE
        )
        
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture line colors
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # White point representing the state
        # Starting point
        t_start = 0.4
        point = Dot(axes.c2p(2 * np.cos(t_start), 1.2 * np.sin(t_start)), color=WHITE, radius=0.08)
        
        self.play(FadeIn(point))
        self.wait(0.5)
        
        # Simulate jumps (collisions) to new points on the ellipse
        jumps = [1.5, 2.8, 4.1, 5.5]
        for t_val in jumps:
            new_pos = axes.c2p(2 * np.cos(t_val), 1.2 * np.sin(t_val))
            self.play(point.animate.move_to(new_pos), run_time=0.7)
            self.wait(0.3)
            
        self.wait(2)
