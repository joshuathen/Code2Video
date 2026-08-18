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
        self.setup_layout("Visualizing the Density Change", [
            "Derivatives greater than one expand the number line space.",
            "Values between zero and one compress the space.",
            "Negative derivatives flip the line while scaling it."
        ])

        # Colors
        COLOR_EXPAND = "#FFFF00"  # Yellow
        COLOR_COMPRESS = "#00FF00" # Green
        COLOR_FLIP = "#FF0000"    # Red
        COLOR_INPUT = WHITE
        COLOR_OUTPUT = BLUE_A

        # Coordinate calculation helpers
        # val from 0 to 5 (mapping to cols 1 to 6)
        def get_pos(row, val): 
            start = self.grid[f"{row}1"]
            end = self.grid[f"{row}6"]
            return start + (val / 5.0) * (end - start)

        # Static Elements: Two Number Lines (Col 2 to Col 6)
        input_line = Line(self.grid["B2"], self.grid["B6"], color=COLOR_INPUT)
        output_line = Line(self.grid["D2"], self.grid["D6"], color=COLOR_OUTPUT)
        
        # Labels at Col 1
        input_label = Text("Input x", font_size=18, color=COLOR_INPUT)
        self.place_at_grid(input_label, "B1", scale_factor=0.6)
        
        output_label = Text("Output f(x)", font_size=18, color=COLOR_OUTPUT)
        self.place_at_grid(output_label, "D1", scale_factor=0.6)

        self.add(input_line, output_line, input_label, output_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_EXPAND)

        # Expand: x in [1.2, 1.6, 2.0] -> y in [1.2, 2.0, 2.8] (f'(x) = 2.0)
        in_pts_exp = VGroup(*[Dot(get_pos("B", x), radius=0.06, color=COLOR_EXPAND) for x in [1.2, 1.6, 2.0]])
        out_pts_exp = VGroup(*[Dot(get_pos("D", y), radius=0.06, color=COLOR_EXPAND) for y in [1.2, 2.0, 2.8]])
        
        arrows_exp = VGroup(*[
            Arrow(in_pts_exp[i].get_center(), out_pts_exp[i].get_center(), 
                  buff=0.1, stroke_width=2, color=COLOR_EXPAND)
            for i in range(3)
        ])
        
        expand_text = Text("Expand (f' > 1)", font_size=16, color=COLOR_EXPAND)
        # Issue 29 Fix: Move to C1 and scale down
        self.place_at_grid(expand_text, "C1", scale_factor=0.6)

        self.play(Create(in_pts_exp))
        self.play(Create(arrows_exp), Create(out_pts_exp))
        self.play(Write(expand_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_COMPRESS)

        # Compress: x in [2.5, 3.0, 3.5] -> y in [3.0, 3.25, 3.5] (f'(x) = 0.5)
        in_pts_com = VGroup(*[Dot(get_pos("B", x), radius=0.06, color=COLOR_COMPRESS) for x in [2.5, 3.0, 3.5]])
        out_pts_com = VGroup(*[Dot(get_pos("D", y), radius=0.06, color=COLOR_COMPRESS) for y in [3.0, 3.25, 3.5]])
        
        arrows_com = VGroup(*[
            Arrow(in_pts_com[i].get_center(), out_pts_com[i].get_center(), 
                  buff=0.1, stroke_width=2, color=COLOR_COMPRESS)
            for i in range(3)
        ])
        
        compress_text = Text("Compress (0 < f' < 1)", font_size=16, color=COLOR_COMPRESS)
        # Issue 30 Fix: Use place_in_area and scale down
        self.place_in_area(compress_text, "C4", "C5", scale_factor=0.4)

        self.play(Create(in_pts_com))
        self.play(Create(arrows_com), Create(out_pts_com))
        self.play(Write(compress_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_FLIP)

        # Flip: x in [4.2, 4.6, 5.0] -> y in [5.0, 4.6, 4.2] (f'(x) = -1)
        in_pts_flip = VGroup(*[Dot(get_pos("B", x), radius=0.06, color=COLOR_FLIP) for x in [4.2, 4.6, 5.0]])
        out_pts_flip = VGroup(*[Dot(get_pos("D", y), radius=0.06, color=COLOR_FLIP) for y in [5.0, 4.6, 4.2]])
        
        arrows_flip = VGroup(*[
            Arrow(in_pts_flip[i].get_center(), out_pts_flip[i].get_center(), 
                  buff=0.1, stroke_width=2, color=COLOR_FLIP)
            for i in range(3)
        ])
        
        flip_text = Text("Flip (f' < 0)", font_size=16, color=COLOR_FLIP)
        # Issue 31 Fix: Scale down at C6
        self.place_at_grid(flip_text, "C6", scale_factor=0.5)

        self.play(Create(in_pts_flip))
        self.play(Create(arrows_flip), Create(out_pts_flip))
        self.play(Write(flip_text))
        self.wait(2)
