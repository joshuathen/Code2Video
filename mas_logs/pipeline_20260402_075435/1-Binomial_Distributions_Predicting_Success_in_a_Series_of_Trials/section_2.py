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
        # Colors
        CYAN = "#00FFFF"
        YELLOW_BULB = "#FFFF00"
        GREY_BULB = "#808080"
        GREEN_IND = "#90EE90"
        ORANGE_N = "#FFA500"
        PINK_P = "#FFC0CB"
        BULB_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/lightbulb.svg"

        self.setup_layout(
            "The Four Conditions (The BINS Criteria)", 
            [
                "To use a binomial distribution, check four criteria.",
                "First: Is it Binary? Success or Failure only.",
                "Second: Are trials Independent? One doesn't affect another.",
                "Third: Is the Number of trials fixed?",
                "Fourth: Is the Success probability the same each time?"
            ]
        )
        
        # Helper to create a lightbulb using the specified asset
        def get_bulb(color=GREY_BULB):
            return SVGMobject(BULB_ASSET).set_color(color)

        # === Animation for Lecture Line 1 ===
        # Vertical letters B, I, N, S appear in cyan (#00FFFF).
        self.lecture[0].set_color(CYAN)
        b_label = Text("B", color=CYAN, weight=BOLD)
        i_label = Text("I", color=CYAN, weight=BOLD)
        n_label = Text("N", color=CYAN, weight=BOLD)
        s_label = Text("S", color=CYAN, weight=BOLD)

        self.place_at_grid(b_label, "B1", scale_factor=0.8)
        self.place_at_grid(i_label, "C1", scale_factor=0.8)
        self.place_at_grid(n_label, "D1", scale_factor=0.8)
        self.place_at_grid(s_label, "E1", scale_factor=0.8)

        bins_vgroup = VGroup(b_label, i_label, n_label, s_label)
        self.play(Write(bins_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # B: Binary - A lightbulb icon toggles yellow/grey (#FFFF00/#808080).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW_BULB)
        
        bulb_binary = get_bulb(GREY_BULB)
        self.place_at_grid(bulb_binary, "B2", scale_factor=0.8)
        binary_text = Text("Binary", font_size=20, color=YELLOW_BULB).next_to(bulb_binary, RIGHT, buff=0.3)
        
        self.play(FadeIn(bulb_binary), FadeIn(binary_text))
        # Toggle
        self.play(bulb_binary.animate.set_color(YELLOW_BULB), run_time=0.5)
        self.play(bulb_binary.animate.set_color(GREY_BULB), run_time=0.5)
        self.play(bulb_binary.animate.set_color(YELLOW_BULB), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # I: Independent - Two lightbulbs flash independently, no causal link.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN_IND)
        
        bulb_i1 = get_bulb(GREY_BULB)
        bulb_i2 = get_bulb(GREY_BULB)
        self.place_at_grid(bulb_i1, "C2", scale_factor=0.7)
        self.place_at_grid(bulb_i2, "C3", scale_factor=0.7)
        ind_text = Text("Independent", font_size=20, color=GREEN_IND).next_to(bulb_i2, RIGHT, buff=0.3)
        
        self.play(FadeIn(bulb_i1), FadeIn(bulb_i2), FadeIn(ind_text))
        
        # Flashing independently
        self.play(bulb_i1.animate.set_color(YELLOW_BULB), run_time=0.3)
        self.play(bulb_i1.animate.set_color(GREY_BULB), bulb_i2.animate.set_color(YELLOW_BULB), run_time=0.4)
        self.play(bulb_i2.animate.set_color(GREY_BULB), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # N: Number - Row of 10 bulbs appears, labeled 'n=10'.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(ORANGE_N)
        
        bulbs_n = VGroup(*[get_bulb(GREY_BULB).scale(0.3) for _ in range(10)]).arrange(RIGHT, buff=0.1)
        # Resolved Issue 34: Move bulbs_n area to D3-D5 and scale to 0.7
        self.place_in_area(bulbs_n, "D3", "D5", scale_factor=0.7)
        
        n_label_text = Text("n = 10", font_size=24, color=ORANGE_N)
        # Resolved Issue 33: Move n_label_text to D2 and scale to 0.8
        self.place_at_grid(n_label_text, "D2", scale_factor=0.8)
        
        self.play(LaggedStart(*[FadeIn(b) for b in bulbs_n], lag_ratio=0.1))
        self.play(Write(n_label_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # S: Same probability - Each bulb shows 'p=0.05' underneath.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PINK_P)
        
        prob_labels = VGroup(*[
            Text("p=0.05", font_size=12, color=PINK_P).next_to(bulbs_n[i], DOWN, buff=0.1) 
            for i in [0, 2, 4, 6, 8] # Just a few to avoid clutter
        ])
        s_text = Text("Same Prob.", font_size=20, color=PINK_P)
        # Resolved Issue 32: Move s_text to E2 and scale to 0.8
        self.place_at_grid(s_text, "E2", scale_factor=0.8)
        
        self.play(FadeIn(prob_labels), Write(s_text))
        
        # Final highlight of all criteria
        self.play(
            *[line.animate.set_color(WHITE) for line in self.lecture],
            bins_vgroup.animate.scale(1.1).set_color(WHITE),
            run_time=2
        )
        self.wait(2)
