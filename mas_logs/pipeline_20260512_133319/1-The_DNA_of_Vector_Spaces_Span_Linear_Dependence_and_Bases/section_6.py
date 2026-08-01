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
        # Initialize the layout with updated script lines
        lecture_lines = [
            "Linear combinations and span define our reachable space.",
            "An independent basis provides the most efficient toolkit.",
            "Mastering these foundations unlocks all of linear algebra."
        ]
        self.setup_layout("Summary & Synthesis", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Line color matches summary text
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        word_span = Text("Span", color="#FFFFFF")
        word_ind = Text("Independence", color="#FFFFFF")
        word_basis = Text("Basis", color="#FFFFFF")
        
        # Position words in a column on the right
        self.place_at_grid(word_span, "B4", scale_factor=0.9)
        self.place_at_grid(word_ind, "C4", scale_factor=0.7)  # Resolved Issue 42
        self.place_at_grid(word_basis, "D4", scale_factor=0.9)
        
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/toolkit.svg]
        toolkit_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/toolkit.svg", color=WHITE)
        self.place_at_grid(toolkit_icon, "C5", scale_factor=0.6) # Resolved Issue 26
        
        summary_group = VGroup(word_span, word_ind, word_basis, toolkit_icon)
        self.play(Write(summary_group))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line color matches the primary basis vector
        self.play(
            self.lecture[1].animate.set_color("#00FF00"),
            FadeOut(summary_group)
        )
        
        # Visualizing a basis generating a span (grid)
        # Choose a central origin on the right grid
        origin_point = self.grid["D4"]
        
        # Define basis vectors
        v1_vec = np.array([1.2, 0, 0])
        v2_vec = np.array([0.4, 0.8, 0])
        
        v1 = Arrow(start=origin_point, end=origin_point + v1_vec, color="#00FF00", buff=0)
        v2 = Arrow(start=origin_point, end=origin_point + v2_vec, color="#0000FF", buff=0)
        
        # Generate a parallelogram grid
        grid_lines = VGroup()
        # Lines parallel to v2
        for i in range(-3, 4):
            line_start = origin_point + i * v1_vec - 2 * v2_vec
            line_end = origin_point + i * v1_vec + 2 * v2_vec
            grid_lines.add(Line(line_start, line_end, color=GREY, stroke_opacity=0.4))
        
        # Lines parallel to v1
        for j in range(-2, 3):
            line_start = origin_point + j * v2_vec - 3 * v1_vec
            line_end = origin_point + j * v2_vec + 3 * v1_vec
            grid_lines.add(Line(line_start, line_end, color=GREY, stroke_opacity=0.4))

        self.play(GrowArrow(v1), GrowArrow(v2))
        self.play(Create(grid_lines, run_time=2))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Line color matches final title
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            FadeOut(grid_lines),
            FadeOut(v1),
            FadeOut(v2)
        )
        
        final_title = Text("The DNA of Vector Spaces", color="#FFFF00")
        # Resolved Issue 40 and 41: Adjusted position and scale
        self.place_in_area(final_title, "C2", "E6", scale_factor=0.8)
        
        self.play(Write(final_title))
        # Pulse effect: scale up and down
        self.play(
            final_title.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
