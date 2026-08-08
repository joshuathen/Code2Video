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
        self.setup_layout("The Phase Space Visualization", [
            "Phase space diagrams explain this pattern.",
            "Velocities reflect within a circular arc.",
            "Reflections trace an arc length."
        ])
        
        # Prepare objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True}, x_length=3, y_length=3)
        axes_labels = axes.get_axis_labels(x_label="v_1", y_label="v_2")
        arc = Arc(radius=1.2, start_angle=0, angle=PI/2, arc_center=axes.c2p(0, 0))
        arc.set_color(BLUE)
        
        # Asset integration (using a default dot as none.svg is empty/placeholder)
        dot = Dot(color=YELLOW)
        dot_label = Text("Point", font_size=16, color=YELLOW)
        
        velocity_diagram = VGroup(axes, axes_labels, arc)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(velocity_diagram, 'A4', 'D6', scale_factor=0.9)
        self.play(Create(axes), Write(axes_labels))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.play(Create(arc))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Animate bouncing point
        self.place_at_grid(dot, 'D4', scale_factor=1.0)
        self.place_at_grid(dot_label, 'E5', scale_factor=0.7)
        self.add(dot, dot_label)
        self.play(MoveAlongPath(dot, arc), run_time=3, rate_func=linear)
        self.wait(1)
