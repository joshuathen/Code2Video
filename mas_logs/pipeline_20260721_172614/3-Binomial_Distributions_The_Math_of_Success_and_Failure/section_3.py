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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The 'BINS' Checklist"
        lines = [
            "Use the BINS acronym to check for binomial distributions.",
            "B stands for Binary: outcomes are success or failure.",
            "I means trials are independent of each other.",
            "N requires a fixed number of trials.",
            "S means the probability remains the same every time."
        ]
        
        self.setup_layout(title, lines)

        # Colors for the BINS acronym
        color_b = "#00FFFF"  # Cyan
        color_i = "#FFFF00"  # Yellow
        color_n = "#00FF00"  # Green
        color_s = "#FFA500"  # Orange

        # Assets
        dice_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg"
        counter_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/counter.svg"

        # === Animation for Lecture Line 1 ===
        # Use the BINS acronym to check for binomial distributions.
        
        b_txt = Text("B", font_size=72, color=color_b)
        i_txt = Text("I", font_size=72, color=color_i)
        n_txt = Text("N", font_size=72, color=color_n)
        s_txt = Text("S", font_size=72, color=color_s)
        
        bins_group = VGroup(b_txt, i_txt, n_txt, s_txt).arrange(RIGHT, buff=0.5)
        # Fix: Issue 28 - Position bins_group at A3-B6
        self.place_in_area(bins_group, "A3", "B6")
        
        self.play(Write(bins_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # B stands for Binary: outcomes are success or failure.
        
        self.play(self.lecture[1].animate.set_color(color_b))
        
        # Highlight 'B' by making others dim
        self.play(
            i_txt.animate.set_opacity(0.3),
            n_txt.animate.set_opacity(0.3),
            s_txt.animate.set_opacity(0.3),
            b_txt.animate.scale(1.2)
        )
        
        # Binary toggle: Success / Failure
        s_label = Text("Success", font_size=24, color=GREEN)
        f_label = Text("Failure", font_size=24, color=RED)
        binary_vg = VGroup(s_label, f_label).arrange(RIGHT, buff=1)
        self.place_in_area(binary_vg, "C3", "C6")
        
        self.play(Create(binary_vg))
        # Visual toggle effect
        self.play(s_label.animate.scale(1.2), f_label.animate.set_opacity(0.5), run_time=0.5)
        self.play(s_label.animate.scale(1/1.2).set_opacity(0.5), f_label.animate.scale(1.2).set_opacity(1), run_time=0.5)
        self.play(s_label.animate.set_opacity(1), f_label.animate.scale(1/1.2).set_opacity(1), run_time=0.5)
        
        self.wait(1)
        self.play(FadeOut(binary_vg))

        # === Animation for Lecture Line 3 ===
        # I means trials are independent of each other.
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_i),
            b_txt.animate.set_opacity(0.3).scale(1/1.2),
            i_txt.animate.set_opacity(1).scale(1.2)
        )
        
        # Issue 21: Use dice.svg asset
        dice1 = SVGMobject(dice_asset_path).set_color(WHITE)
        dice2 = SVGMobject(dice_asset_path).set_color(WHITE)
        dice_vg = VGroup(dice1, dice2).arrange(RIGHT, buff=1)
        self.place_in_area(dice_vg, "C3", "C6", scale_factor=0.8)
        
        self.play(Create(dice_vg))
        # Roll them independently
        self.play(
            Rotate(dice1, angle=2*PI),
            Rotate(dice2, angle=-2*PI),
            run_time=1
        )
        
        self.wait(1)
        self.play(FadeOut(dice_vg))

        # === Animation for Lecture Line 4 ===
        # N requires a fixed number of trials.
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(color_n),
            i_txt.animate.set_opacity(0.3).scale(1/1.2),
            n_txt.animate.set_opacity(1).scale(1.2)
        )
        
        # Issue 21: Use counter.svg asset
        counter_icon = SVGMobject(counter_asset_path).set_color(color_n)
        
        # Issue 29: Position n_label at C3-C6, scale 0.8
        n_val = ValueTracker(0)
        n_display = DecimalNumber(0, num_decimal_places=0, font_size=48, color=color_n)
        n_display.add_updater(lambda d: d.set_value(n_val.get_value()))
        
        # Group counter icon and display
        n_label = VGroup(counter_icon, n_display).arrange(RIGHT, buff=0.3)
        self.place_in_area(n_label, "C3", "C6", scale_factor=0.8)
        
        # Issue 29: Position target_n at D4
        target_n = Text("Target: n=5", font_size=24, color=WHITE)
        self.place_at_grid(target_n, "D4")
        
        self.play(FadeIn(n_label), Write(target_n))
        self.play(n_val.animate.set_value(5), run_time=2, rate_func=linear)
        
        self.wait(1)
        self.play(FadeOut(n_label), FadeOut(target_n))

        # === Animation for Lecture Line 5 ===
        # S means the probability remains the same every time.
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(color_s),
            n_txt.animate.set_opacity(0.3).scale(1/1.2),
            s_txt.animate.set_opacity(1).scale(1.2)
        )
        
        # Issue 30: Position p_label at C3-C6, scale 0.8
        p_label = Text("p = 0.6", font_size=48, color=color_s)
        self.place_in_area(p_label, "C3", "C6", scale_factor=0.8)
        
        # Issue 30: Position consistency_txt at D4
        consistency_txt = Text("Constant", font_size=24, color=WHITE)
        self.place_at_grid(consistency_txt, "D4")
        
        self.play(Write(p_label), Write(consistency_txt))
        
        self.play(p_label.animate.scale(1.1), run_time=0.5)
        self.play(p_label.animate.scale(1/1.1), run_time=0.5)
        
        self.wait(1)
        
        # Reset acronym to full visibility
        self.play(
            self.lecture[4].animate.set_color(WHITE),
            b_txt.animate.set_opacity(1),
            i_txt.animate.set_opacity(1),
            n_txt.animate.set_opacity(1),
            s_txt.animate.scale(1/1.2),
            FadeOut(p_label),
            FadeOut(consistency_txt)
        )
        self.wait(2)
