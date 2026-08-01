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
        # Setup title and lecture lines
        title_str = "Eigenvectors: The Unshakable Lines"
        lines_str = [
            "Some lines stay on their span during transformation.",
            "These special directions are called eigenvectors.",
            "They only get stretched or squished, never rotated.",
            "The stretching factor is known as the eigenvalue.",
            "They reveal the fundamental axis of the transformation."
        ]
        self.setup_layout(title_str, lines_str)

        # Colors
        GREY = "#D3D3D3"
        YELLOW = "#FFFF00"
        CYAN = "#00FFFF"
        MAGENTA = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # Color line 1 grey
        self.lecture[0].set_color(GREY)
        
        # Define origin point within the grid area
        origin_pt = Dot(point=self.grid["D3"], radius=0.05, color=WHITE)
        self.add(origin_pt)
        
        # Radiating grey vectors
        vec_coords = [[1, 0], [0, 1], [0.7, 0.7], [-0.7, 0.7], [-1, 0], [0, -1]]
        grey_vectors = VGroup(*[
            Arrow(start=origin_pt.get_center(), 
                  end=origin_pt.get_center() + np.array([c[0], c[1], 0]), 
                  buff=0, color=GREY, stroke_width=4)
            for c in vec_coords
        ])
        
        self.play(Create(grey_vectors))
        self.wait(1)
        
        # Apply transformation: Horizontal stretch (Matrix [[1.5, 0], [0, 1]])
        transformed_coords = [[c[0]*1.5, c[1]] for c in vec_coords]
        
        self.play(
            *[
                grey_vectors[i].animate.put_start_and_end_on(
                    origin_pt.get_center(), 
                    origin_pt.get_center() + np.array([transformed_coords[i][0], transformed_coords[i][1], 0])
                )
                for i in range(len(grey_vectors))
            ],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2 yellow
        self.lecture[1].set_color(YELLOW)
        
        # Highlight horizontal eigenvector (index 0)
        eigen_vec = grey_vectors[0]
        self.play(eigen_vec.animate.set_color(YELLOW).set_stroke_width(6))
        
        eigen_label = Text("Eigenvector", font_size=18, color=YELLOW)
        # Resolved Issue 40: Scale to 0.8
        self.place_at_grid(eigen_label, "C5", scale_factor=0.8)
        self.play(Write(eigen_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3 yellow
        self.lecture[2].set_color(YELLOW)
        
        # Emphasize stretching/squishing along the span
        self.play(
            eigen_vec.animate.put_start_and_end_on(origin_pt.get_center(), origin_pt.get_center() + np.array([2.5, 0, 0])),
            run_time=1.0
        )
        self.play(
            eigen_vec.animate.put_start_and_end_on(origin_pt.get_center(), origin_pt.get_center() + np.array([0.5, 0, 0])),
            run_time=1.0
        )
        self.play(
            eigen_vec.animate.put_start_and_end_on(origin_pt.get_center(), origin_pt.get_center() + np.array([1.5, 0, 0])),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color line 4 cyan
        self.lecture[3].set_color(CYAN)
        
        # Stretching factor = Eigenvalue
        val_label = MathTex(r"\lambda = 1.5", font_size=24, color=CYAN)
        # Resolved Issue 39: Use place_in_area
        self.place_in_area(val_label, 'E4', 'E6', scale_factor=0.8)
        self.play(Write(val_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Color line 5 magenta
        self.lecture[4].set_color(MAGENTA)
        
        # Contrast with off-axis vector (index 2: [0.7, 0.7])
        off_axis_vec = grey_vectors[2]
        
        # Show its original span line
        span_line = Line(origin_pt.get_center() - np.array([1.5, 1.5, 0]), 
                         origin_pt.get_center() + np.array([1.5, 1.5, 0]), 
                         color=GREY, stroke_width=1, stroke_opacity=0.3)
        self.play(Create(span_line))
        
        self.play(off_axis_vec.animate.set_color(MAGENTA))
        
        # Rewind off-axis to original span to show rotation clearly
        self.play(
            off_axis_vec.animate.put_start_and_end_on(origin_pt.get_center(), origin_pt.get_center() + np.array([0.7, 0.7, 0])),
            run_time=1
        )
        self.wait(0.5)
        # Apply transformation again specifically for this vector
        self.play(
            off_axis_vec.animate.put_start_and_end_on(origin_pt.get_center(), origin_pt.get_center() + np.array([1.05, 0.7, 0])),
            run_time=2
        )
        
        rot_label = Text("Rotates off span", font_size=16, color=MAGENTA)
        # Resolved Issue 38: Use place_in_area
        self.place_in_area(rot_label, 'B3', 'B5', scale_factor=0.8)
        self.play(Write(rot_label))
        self.wait(2)
