from manim import *
import numpy as np

# === TeachingScene Base Class ===
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
        # Data
        title = "Convergence: When Does an Infinite Sum Stop?"
        lines = [
            "Sums converge when their terms shrink toward zero.",
            "In this world, terms shrink if they gain more 2s.",
            "1 + 2 + 4 + 8 grows normally but 2-adically shrinks."
        ]
        
        # Setup
        self.setup_layout(title, lines)

        # Colors
        color1 = WHITE
        color2 = "#ADD8E6"
        color3 = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        # Sequence 1, 2, 4, 8...
        seq_text = MathTex("1, 2, 4, 8, 16, \\dots", color=color1)
        # Issue 30: Move seq_text to A3
        self.place_at_grid(seq_text, "A3", scale_factor=0.9)
        
        cond = MathTex("a_n \\to 0?", color=color1)
        # Issue 30: Move cond to A5
        self.place_at_grid(cond, "A5", scale_factor=0.9)
        
        self.play(Write(seq_text))
        self.wait(0.5)
        self.play(Write(cond))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        adic_val = MathTex("|2^n|_2 = 2^{-n} \\to 0", color=color2)
        # Issue 31: place_in_area B3 to B5
        self.place_in_area(adic_val, "B3", "B5", scale_factor=1.0)
        
        self.play(Write(adic_val))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Ruler Layout
        std_label = Text("Standard World", font_size=16, color=color1)
        self.place_at_grid(std_label, "C1", scale_factor=0.8)
        
        std_line = NumberLine(x_range=[0, 16, 4], length=4, include_numbers=True, font_size=12, color=color1)
        self.place_in_area(std_line, "C2", "C6", scale_factor=1.0)
        
        adic_label = Text("2-adic World", font_size=16, color=color2)
        self.place_at_grid(adic_label, "E1", scale_factor=0.8)
        
        # 2-adic line for visual representation of steps shrinking
        adic_line = NumberLine(x_range=[0, 2, 0.5], length=4, include_numbers=False, color=color2)
        self.place_in_area(adic_line, "E2", "E6", scale_factor=1.0)
        
        self.play(
            FadeOut(seq_text), FadeOut(cond), FadeOut(adic_val),
            FadeIn(std_label), FadeIn(std_line),
            FadeIn(adic_label), FadeIn(adic_line)
        )

        traveler_std = Dot(color=color3, radius=0.1)
        traveler_std.move_to(std_line.n2p(0))
        
        traveler_adic = Dot(color=color3, radius=0.1)
        traveler_adic.move_to(adic_line.n2p(0))
        
        self.add(traveler_std, traveler_adic)

        # Steps
        steps = [1, 2, 4, 8]
        # In 2-adic representation, the visual steps shrink
        visual_adic_steps = [1.0, 0.5, 0.25, 0.125]
        
        curr_std = 0
        curr_adic = 0
        
        for s, ads in zip(steps, visual_adic_steps):
            # Arcs
            arc_std = ArcBetweenPoints(std_line.n2p(curr_std), std_line.n2p(curr_std + s), angle=-PI/2, color=color3, stroke_width=2)
            arc_adic = ArcBetweenPoints(adic_line.n2p(curr_adic), adic_line.n2p(curr_adic + ads), angle=-PI/2, color=color3, stroke_width=2)
            
            self.play(
                Create(arc_std),
                Create(arc_adic),
                traveler_std.animate.move_to(std_line.n2p(curr_std + s)),
                traveler_adic.animate.move_to(adic_line.n2p(curr_adic + ads)),
                run_time=0.8
            )
            curr_std += s
            curr_adic += ads

        # Highlight emphasis
        self.play(Indicate(traveler_adic, color=color3), Flash(traveler_adic, color=color3))
        
        # Show convergence message
        conv_msg = Text("Steps shrink!", font_size=20, color=color3)
        # Issue 32: place_in_area F2 to F5
        self.place_in_area(conv_msg, "F2", "F5", scale_factor=1.0)
        self.play(Write(conv_msg))
        
        self.wait(2)
