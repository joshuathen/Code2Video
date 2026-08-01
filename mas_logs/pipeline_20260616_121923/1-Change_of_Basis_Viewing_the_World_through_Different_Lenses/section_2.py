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
        # Define Colors
        COLOR_BLUE = "#87CEEB"
        COLOR_GOLD = "#FFD700"
        COLOR_WHITE = "#FFFFFF"
        COLOR_GREY = "#666666"

        # Initialize layout
        self.setup_layout(
            "Prerequisite Check: What is a Basis?",
            [
                'A basis provides building blocks for any point.',
                'Usually, we use the standard blue grid.',
                'But any set of independent vectors works.'
            ]
        )

        # Common coordinate plane for right-side visual area
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": GREY, "stroke_opacity": 0.3}
        )
        self.place_in_area(plane, "B2", "E5")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_BLUE))
        
        i_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=COLOR_BLUE)
        j_vec = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=COLOR_BLUE)
        
        # Grid positioning for labels to avoid manual .next_to or overlaps
        i_label = Text("î", color=COLOR_BLUE, font_size=20)
        self.place_at_grid(i_label, 'D4', scale_factor=0.8)
        
        j_label = Text("ĵ", color=COLOR_BLUE, font_size=20)
        self.place_at_grid(j_label, 'D3', scale_factor=0.6) # Fixed per Issue 44
        
        self.play(Create(plane))
        self.play(
            GrowArrow(i_vec), 
            GrowArrow(j_vec), 
            Write(i_label), 
            Write(j_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(COLOR_GREY),
            self.lecture[1].animate.set_color(COLOR_GOLD)
        )
        
        # Basis vectors: b1 = (2, 1), b2 = (-1, 1)
        b1_vec = Arrow(plane.c2p(0, 0), plane.c2p(2, 1), buff=0, color=COLOR_GOLD)
        b2_vec = Arrow(plane.c2p(0, 0), plane.c2p(-1, 1), buff=0, color=COLOR_GOLD)
        
        b1_label = Text("b₁", color=COLOR_GOLD, font_size=20)
        self.place_at_grid(b1_label, 'C5', scale_factor=0.7) # Fixed per Issue 46
        
        b2_label = Text("b₂", color=COLOR_GOLD, font_size=20)
        self.place_at_grid(b2_label, 'B3', scale_factor=0.7)
        
        self.play(
            GrowArrow(b1_vec), 
            GrowArrow(b2_vec), 
            Write(b1_label), 
            Write(b2_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(COLOR_GREY),
            self.lecture[2].animate.set_color(COLOR_WHITE)
        )
        
        # Point P at (1, 2)
        p_pos = plane.c2p(1, 2)
        p_dot = Dot(p_pos, color=COLOR_WHITE)
        
        p_label = Text("P", color=COLOR_WHITE, font_size=24, slant=ITALIC)
        self.place_at_grid(p_label, 'B5', scale_factor=0.8) # Fixed per Issue 45
        
        self.play(Create(p_dot), Write(p_label))
        self.wait(0.5)
        
        # Highlight standard basis sum: P = 1*i + 2*j
        i_comp = Line(plane.c2p(0,0), plane.c2p(1,0), color=COLOR_BLUE, stroke_width=6)
        j_comp = Line(plane.c2p(1,0), plane.c2p(1,2), color=COLOR_BLUE, stroke_width=6)
        
        self.play(Create(i_comp))
        self.play(Create(j_comp))
        self.wait(1)
        self.play(FadeOut(i_comp), FadeOut(j_comp))
        
        # Highlight gold basis sum: P = 1*b1 + 1*b2
        b1_comp = Line(plane.c2p(0,0), plane.c2p(2,1), color=COLOR_GOLD, stroke_width=6)
        b2_comp = Line(plane.c2p(2,1), plane.c2p(1,2), color=COLOR_GOLD, stroke_width=6)
        
        self.play(Create(b1_comp))
        self.play(Create(b2_comp))
        self.wait(2)
