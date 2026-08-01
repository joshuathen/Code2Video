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
        # Updated lecture lines to match prompt requirements
        lines = [
            'Use two independent vectors to build a space.',
            'These vectors span every single point in 2D.',
            'With no redundancy, every vector provides unique direction.',
            'This efficient set is called a basis.',
            'Adding more vectors creates unnecessary linear dependence.'
        ]
        self.setup_layout("The Basis: The Efficient Blueprint", lines)
        
        # Common origin for the visual space
        origin_pos = self.grid["D3"]
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Vectors i and j
        vec_i = Arrow(start=origin_pos, end=origin_pos + RIGHT, buff=0, color="#FF0000", stroke_width=4)
        vec_j = Arrow(start=origin_pos, end=origin_pos + UP, buff=0, color="#00FF00", stroke_width=4)
        
        label_i = Text("i", font_size=18, color="#FFFFFF")
        label_j = Text("j", font_size=18, color="#FFFFFF")
        
        # Position labels at grid cells relative to tips (Issue 37)
        self.place_at_grid(label_i, "E4", scale_factor=1.0)
        self.place_at_grid(label_j, "C4", scale_factor=1.0)
        
        self.play(GrowArrow(vec_i), GrowArrow(vec_j))
        self.play(FadeIn(label_i), FadeIn(label_j))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Coordinate grid based on i and j
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": "#444444",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            }
        )
        self.place_at_grid(plane, "D3")
        
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Flash vectors to signify independence
        self.play(
            Flash(vec_i, color=WHITE, line_length=0.3),
            Flash(vec_j, color=WHITE, line_length=0.3),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Label basis (Issue 39)
        basis_label = Text("Basis for 2D Space", font_size=24, color="#FFFFFF")
        self.place_in_area(basis_label, 'F2', 'F5', scale_factor=1.0)
        
        self.play(Write(basis_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Add a third vector k to show dependence (Issue 38)
        vec_k = Arrow(start=origin_pos, end=origin_pos + RIGHT + UP, buff=0, color="#0000FF", stroke_width=4)
        label_k = Text("k", font_size=18, color="#0000FF")
        self.place_at_grid(label_k, 'B5', scale_factor=1.0)
        
        # Visual proof of dependence (k = i + j)
        ghost_i = Arrow(start=origin_pos, end=origin_pos + RIGHT, buff=0, color="#FF0000", stroke_width=2).set_opacity(0.5)
        ghost_j = Arrow(start=origin_pos + RIGHT, end=origin_pos + RIGHT + UP, buff=0, color="#00FF00", stroke_width=2).set_opacity(0.5)
        
        self.play(GrowArrow(vec_k), FadeIn(label_k))
        self.play(FadeIn(ghost_i), FadeIn(ghost_j))
        self.wait(0.5)
        
        redundancy_text = Text("k = i + j (Redundant)", font_size=18, color="#0000FF")
        self.place_at_grid(redundancy_text, "A3", scale_factor=1.0)
        
        self.play(Write(redundancy_text))
        
        # Mark k as redundant
        cross_mark = Text("X", color=RED, font_size=40).move_to(vec_k.get_center())
        self.play(FadeIn(cross_mark))
        self.wait(2)
