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
        lecture_lines = [
            "Imagine two parallel lines for input and output.",
            "Connect inputs to outputs using mapping arrows.",
            "For f(x) equals 3x, arrows spread out wider.",
            "The output space stretches relative to the input.",
            "This mapping view shows how functions transform space."
        ]
        self.setup_layout("Prerequisite: The Mapping View", lecture_lines)

        def val_to_pos(val, row_char):
            # Internal mapping logic to keep things on the grid
            # val 0 is at Col 2 (x=1.5)
            # 1 unit of value = 0.5 units of grid x
            x_coord = 1.5 + val * 0.5
            y_coord = self.grid[f"{row_char}1"][1]
            return np.array([x_coord, y_coord, 0])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Parallel lines from Col 2 to Col 6
        input_line = Line(self.grid["B2"], self.grid["B6"], color=WHITE)
        output_line = Line(self.grid["E2"], self.grid["E6"], color=WHITE)
        
        input_label = Text("Input x", font_size=18, color=WHITE)
        output_label = Text("Output f(x)", font_size=18, color=WHITE)
        
        # Fixes for issues 27, 28, 29: Use place_in_area with scale_factor 0.8
        self.place_in_area(input_label, "A2", "A3", scale_factor=0.8)
        self.place_in_area(output_label, "D2", "D3", scale_factor=0.8)

        self.play(Create(input_line), Create(output_line), Write(input_label), Write(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        grey_arrows = VGroup()
        for v in range(9): # v=0 to 8
            start = val_to_pos(v, "B")
            end = val_to_pos(v, "E")
            arrow = Arrow(start, end, color=GREY, stroke_width=1, buff=0.05)
            grey_arrows.add(arrow)
            
        self.play(Create(grey_arrows))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Specific mapping for f(x)=3x
        # x=1 -> f(x)=3
        # x=2 -> f(x)=6
        p_in_1 = val_to_pos(1, "B")
        p_in_2 = val_to_pos(2, "B")
        p_out_1 = val_to_pos(3, "E")
        p_out_2 = val_to_pos(6, "E")
        
        cyan_arrow1 = Arrow(p_in_1, p_out_1, color="#00FFFF", buff=0.1)
        cyan_arrow2 = Arrow(p_in_2, p_out_2, color="#00FFFF", buff=0.1)
        
        label_1 = Text("1", font_size=16).next_to(p_in_1, UP, buff=0.1)
        label_2 = Text("2", font_size=16).next_to(p_in_2, UP, buff=0.1)
        label_3 = Text("3", font_size=16).next_to(p_out_1, DOWN, buff=0.1)
        label_6 = Text("6", font_size=16).next_to(p_out_2, DOWN, buff=0.1)
        
        self.play(grey_arrows.animate.set_opacity(0.2))
        self.play(
            GrowArrow(cyan_arrow1), GrowArrow(cyan_arrow2),
            Write(label_1), Write(label_2),
            Write(label_3), Write(label_6)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        input_segment = Line(p_in_1, p_in_2, color=YELLOW, stroke_width=5)
        output_segment = Line(p_out_1, p_out_2, color="#00FF00", stroke_width=5)
        
        brace_in = BraceBetweenPoints(p_in_1, p_in_2, UP, color=YELLOW, buff=0.4)
        brace_out = BraceBetweenPoints(p_out_1, p_out_2, DOWN, color="#00FF00", buff=0.4)
        
        brace_in_text = Text("1 unit", font_size=14, color=YELLOW).next_to(brace_in, UP, buff=0.05)
        brace_out_text = Text("3 units", font_size=14, color="#00FF00").next_to(brace_out, DOWN, buff=0.05)
        
        self.play(Create(input_segment), Create(output_segment))
        self.play(Write(brace_in), Write(brace_in_text), Write(brace_out), Write(brace_out_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Bloomer: many small dots between x=1 and x=2
        num_dots = 11
        bloomers = VGroup(*[Dot(val_to_pos(1 + i/(num_dots-1), "B"), radius=0.03, color=YELLOW) for i in range(num_dots)])
        
        self.play(
            FadeOut(cyan_arrow1), FadeOut(cyan_arrow2),
            FadeOut(brace_in), FadeOut(brace_in_text),
            FadeOut(brace_out), FadeOut(brace_out_text),
            FadeOut(input_segment), FadeOut(output_segment),
            FadeOut(label_1), FadeOut(label_2), FadeOut(label_3), FadeOut(label_6)
        )
        
        self.play(Create(bloomers))
        self.play(
            *[bloomers[i].animate.move_to(val_to_pos((1 + i/(num_dots-1)) * 3, "E")) for i in range(num_dots)],
            run_time=2,
            rate_func=slow_into
        )
        self.wait(2)
