from manim import *
import numpy as np
import random

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
        # Section 3: Dirichlet’s Theorem: The Infinite Paths
        title = "Dirichlet’s Theorem: The Infinite Paths"
        lines = [
            "Dirichlet’s formula identifies paths where primes never end.",
            "These lanes stretch infinitely into the number universe.",
            "Both tracks are populated by an endless supply of sparks.",
            "No matter how far we travel, primes remain abundant.",
            "These arithmetic paths contain infinitely many primes."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_BLUE = "#ADD8E6"
        COLOR_ORANGE = "#FFCC99"
        COLOR_GOLD = "#FFD700"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_BLUE)
        
        # Display the formula 'an + b' with 'gcd(a, b) = 1'
        formula = Text("an + b", color=COLOR_WHITE)
        condition = Text("gcd(a, b) = 1", color=COLOR_WHITE)
        formula_group = VGroup(formula, condition).arrange(DOWN, buff=0.2)
        # Resolved Issue #28: Expanded area and updated scale
        self.place_in_area(formula_group, "A1", "B6", scale_factor=0.9)
        
        self.play(Write(formula_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_ORANGE)
        
        # Vanishing point near bottom right area
        v_point = self.grid["E6"] + RIGHT*0.5
        
        # Starting points for two parallel paths (in perspective)
        # Path 1 (Blue)
        p1_top_left = self.grid["C1"]
        p1_top_right = self.grid["C2"]
        path1 = Polygon(p1_top_left, p1_top_right, v_point, v_point, 
                        fill_opacity=0.3, fill_color=COLOR_BLUE, stroke_width=1, stroke_color=COLOR_BLUE)
        
        # Path 2 (Orange)
        p2_top_left = self.grid["C4"]
        p2_top_right = self.grid["C5"]
        path2 = Polygon(p2_top_left, p2_top_right, v_point, v_point, 
                        fill_opacity=0.3, fill_color=COLOR_ORANGE, stroke_width=1, stroke_color=COLOR_ORANGE)

        self.play(Create(path1), Create(path2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_GOLD)
        
        # Many small golden sparks appear randomly along both paths
        sparks = VGroup()
        for _ in range(40):
            # Blue path sparks
            alpha = random.random()
            side = random.uniform(0, 1)
            pos_l = p1_top_left * (1-alpha) + v_point * alpha
            pos_r = p1_top_right * (1-alpha) + v_point * alpha
            pos = pos_l * (1-side) + pos_r * side
            spark = Dot(pos, radius=0.03, color=COLOR_GOLD)
            sparks.add(spark)
            
            # Orange path sparks
            alpha2 = random.random()
            side2 = random.uniform(0, 1)
            pos_l2 = p2_top_left * (1-alpha2) + v_point * alpha2
            pos_r2 = p2_top_right * (1-alpha2) + v_point * alpha2
            pos2 = pos_l2 * (1-side2) + pos_r2 * side2
            spark2 = Dot(pos2, radius=0.03, color=COLOR_GOLD)
            sparks.add(spark2)

        self.play(FadeIn(sparks, lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_BLUE)
        
        # Resolved Issue #22: Integrated SVGMobject Prime Scout
        scout = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/scout.svg")
        scout.set_color(COLOR_GOLD)
        scout.scale(0.2)
        # Position at the start of the blue path
        scout_start = (p1_top_left + p1_top_right) / 2
        scout.move_to(scout_start)
        
        scout.add_updater(lambda m, dt: m.set_opacity(random.uniform(0.7, 1.0))) # Sparkle effect
        
        self.add(scout)
        self.play(
            scout.animate.move_to(v_point).scale(0.1),
            run_time=2,
            rate_func=exponential_decay
        )
        self.remove(scout)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_WHITE)
        
        # 'Infinitely Many Primes' in glowing white
        inf_text = Text("Infinitely Many Primes", font_size=32, color=COLOR_WHITE)
        # Resolved Issue #27: Move to F1-F6 area to avoid occlusion
        self.place_in_area(inf_text, "F1", "F6", scale_factor=0.7)
        
        glow = inf_text.copy().set_stroke(COLOR_WHITE, 8).set_opacity(0.3)
        inf_group = VGroup(glow, inf_text)
        
        self.play(FadeIn(inf_group, scale=1.2))
        self.play(inf_group.animate.scale(1.1), rate_func=there_and_back, run_time=2)
        self.wait(2)
