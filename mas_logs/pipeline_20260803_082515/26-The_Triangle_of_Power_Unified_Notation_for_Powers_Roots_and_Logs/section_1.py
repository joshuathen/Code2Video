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
        self.setup_layout(
            "The Problem of Disjointed Symbols",
            [
                "Traditional math notation can feel like separate languages.",
                "We use different symbols for powers, roots, and logs.",
                "Leo the Lion sees 2^3, root 3 of 8, and log_2(8).",
                "These look different but describe the same relationship.",
                "Let's find a way to unify these three ideas."
            ]
        )
        
        # Colors
        COLOR_POWER = "#FFD700"  # Gold
        COLOR_ROOT = "#1E90FF"   # DodgerBlue
        COLOR_LOG = "#32CD32"    # LimeGreen
        COLOR_HIGHLIGHT = "#FF0000" # Red
        COLOR_NEUTRAL = "#FFFFFF"   # White

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_POWER))
        
        # Power: 2^3 = 8
        exp_power = MathTex("2", "^{3}", "=", "8", color=COLOR_POWER)
        self.place_at_grid(exp_power, "B2", scale_factor=1.0)
        
        # Root: root_3(8) = 2
        exp_root = MathTex(r"\sqrt[3]{8}", "=", "2", color=COLOR_ROOT)
        self.place_at_grid(exp_root, "B4", scale_factor=1.0)
        
        # Log: log_2(8) = 3
        exp_log = MathTex(r"\log_2(8)", "=", "3", color=COLOR_LOG)
        self.place_at_grid(exp_log, "B6", scale_factor=1.0)

        self.play(
            FadeIn(exp_power),
            FadeIn(exp_root),
            FadeIn(exp_log),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_ROOT))
        
        # Lion asset (SVGMobject requires path or default)
        try:
            leo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lion.svg")
        except:
            leo = Circle(color=ORANGE).scale(0.5) # Fallback if asset missing
            
        leo.set_color(COLOR_NEUTRAL)
        self.place_at_grid(leo, "F6", scale_factor=0.8)
        
        q1 = Text("?", color=COLOR_NEUTRAL).scale(0.8).next_to(leo, UL, buff=0.1)
        q2 = Text("?", color=COLOR_NEUTRAL).scale(0.8).next_to(leo, UR, buff=0.1)
        q3 = Text("?", color=COLOR_NEUTRAL).scale(0.8).next_to(leo, UP, buff=0.1)
        questions = VGroup(q1, q2, q3)

        self.play(
            FadeIn(leo),
            Write(questions),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_LOG))
        
        # Highlight operators
        # Indexing components: 
        # exp_power[1] is ^{3}
        # exp_root[0][0] is usually the root symbol
        # exp_log[0][0:3] is 'log'
        self.play(
            exp_power[1].animate.set_color(COLOR_HIGHLIGHT),
            exp_root[0].animate.set_color(COLOR_HIGHLIGHT),
            exp_log[0].animate.set_color(COLOR_HIGHLIGHT),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_NEUTRAL))
        
        # Use simple circles around the main numbers
        circles = VGroup(
            Circle(color=WHITE, stroke_width=2).scale(0.3).move_to(exp_power[0]),
            Circle(color=WHITE, stroke_width=2).scale(0.3).move_to(exp_root[2]),
            Circle(color=WHITE, stroke_width=2).scale(0.3).move_to(exp_log[2])
        )
        
        self.play(Create(circles), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_POWER))
        
        to_fade = VGroup(
            exp_power,
            exp_root,
            exp_log,
            questions,
            circles
        )
        
        num2 = MathTex("2", color=COLOR_NEUTRAL)
        num3 = MathTex("3", color=COLOR_NEUTRAL)
        num8 = MathTex("8", color=COLOR_NEUTRAL)
        
        self.place_at_grid(num2, "D3", scale_factor=1.2)
        self.place_at_grid(num3, "C4", scale_factor=1.2)
        self.place_at_grid(num8, "D5", scale_factor=1.2)

        self.play(
            FadeOut(to_fade),
            FadeIn(num2),
            FadeIn(num3),
            FadeIn(num8),
            run_time=2
        )
        self.wait(2)
