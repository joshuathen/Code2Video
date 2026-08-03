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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Preview of Higher Dimensions", [
            "Derivatives describe how a function reshapes space.",
            "This concept scales to areas and volumes in 3D.",
            "Welcome to the transformational view of calculus."
        ])
        
        # Colors for the lecture lines
        line_colors = ["#FFFF00", "#00BFFF", "#90EE90"] # YELLOW, DEEP SKY BLUE, LIGHT GREEN

        # === Animation for Lecture Line 1 ===
        # Derivatives describe how a function reshapes space.
        self.play(self.lecture[0].animate.set_color(line_colors[0]))
        
        # Show a single grid line being pulled
        one_d_line = Line(LEFT, RIGHT, color=line_colors[0], stroke_width=6)
        self.place_in_area(one_d_line, 'C2', 'C5', scale_factor=1.5)
        
        stretch_tracker = ValueTracker(1.0)
        initial_width = one_d_line.get_width()
        one_d_line.add_updater(lambda m: m.set_width(initial_width * stretch_tracker.get_value(), stretch=True))
        
        self.play(Create(one_d_line))
        self.play(stretch_tracker.animate.set_value(1.8), run_time=1.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This concept scales to areas and volumes in 3D.
        self.play(self.lecture[1].animate.set_color(line_colors[1]))
        
        # Transition the line into a 2D grid fabric
        h_lines = VGroup(*[Line(LEFT*2, RIGHT*2) for _ in range(5)]).arrange(DOWN, buff=0.5)
        v_lines = VGroup(*[Line(UP*2, DOWN*2) for _ in range(5)]).arrange(RIGHT, buff=0.5)
        simple_grid = VGroup(h_lines, v_lines).set_color(line_colors[1])
        self.place_in_area(simple_grid, 'B2', 'E5', scale_factor=0.7)
        
        # Stop updater before transform
        one_d_line.clear_updaters()
        self.play(ReplacementTransform(one_d_line, simple_grid))
        
        # Warp and stretch the 2D grid
        self.play(
            simple_grid.animate.apply_matrix([[1.3, 0.4, 0], [0.1, 1.1, 0], [0, 0, 1]]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Welcome to the transformational view of calculus.
        self.play(self.lecture[2].animate.set_color(line_colors[2]))
        
        # Display 'Next: Jacobians' in light cyan (#E0FFFF)
        next_text = Text("Next: Jacobians", color="#E0FFFF", font_size=32)
        self.place_at_grid(next_text, 'F3', scale_factor=1.0)
        
        # 2D grid expands further
        self.play(
            simple_grid.animate.scale(1.2).set_opacity(0.6),
            Write(next_text),
            run_time=2
        )
        self.wait(2)
