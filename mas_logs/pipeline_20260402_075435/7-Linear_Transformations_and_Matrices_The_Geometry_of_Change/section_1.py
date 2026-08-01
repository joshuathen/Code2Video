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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initializing scene with title and lecture lines
        title_text = "The Starting Line: Basis Vectors"
        lecture_lines = [
            "Start with the standard 2D Cartesian grid.",
            "Meet i-hat, the basis vector for the x-direction.",
            "Meet j-hat, the basis vector for the y-direction.",
            "Any vector, like (3,2), exists within this grid.",
            "It is simply a sum of scaled basis vectors."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#444444")
        
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={
                "stroke_color": "#444444",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"color": "#444444"}
        )
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.9)
        
        self.play(Create(plane), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        
        i_hat = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(1, 0),
            color="#FF0000",
            buff=0
        )
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        i_label = Text("i", slant=ITALIC, color=WHITE)
        self.place_at_grid(i_label, 'D4', scale_factor=0.8)
        
        self.play(GrowArrow(i_hat), Write(i_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        j_hat = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(0, 1),
            color="#00FF00",
            buff=0
        )
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        j_label = Text("j", slant=ITALIC, color=WHITE)
        self.place_at_grid(j_label, 'C3', scale_factor=0.8)
        
        self.play(GrowArrow(j_hat), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        
        v_32 = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(3, 2),
            color="#FFFF00",
            buff=0
        )
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        v_label = Text("(3,2)", color=WHITE)
        self.place_at_grid(v_label, 'A6', scale_factor=0.8)
        
        self.play(GrowArrow(v_32), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        
        # Horizontal components
        x_arrows = VGroup(*[
            Arrow(
                start=plane.coords_to_point(k, 0),
                end=plane.coords_to_point(k+1, 0),
                color="#FF0000",
                buff=0
            ) for k in range(3)
        ])
        
        # Vertical components
        y_arrows = VGroup(*[
            Arrow(
                start=plane.coords_to_point(3, k),
                end=plane.coords_to_point(3, k+1),
                color="#00FF00",
                buff=0
            ) for k in range(2)
        ])
        
        self.play(FadeIn(x_arrows, shift=RIGHT), run_time=1.5)
        self.play(FadeIn(y_arrows, shift=UP), run_time=1.5)
        self.wait(2)
