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
        # Setup reproducibility
        np.random.seed(42)

        # Setup layout
        self.setup_layout(
            "Defining Entropy: The Average Surprise", 
            [
                'Entropy measures the average information gain for a guess.', 
                'High entropy words spread words evenly across many buckets.', 
                'Low entropy words leave most suspects in one pile.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Issue 44: Scale factor 1.2 and area A1-A6
        entropy_formula = Text("H = Average Information", font_size=24, color=WHITE)
        self.place_in_area(entropy_formula, "A1", "A6", scale_factor=1.2)
        
        self.play(Write(entropy_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # SALET label and bars (Issue 33 & 45)
        salet_text = Text("SALET", font_size=18, color=WHITE)
        self.place_at_grid(salet_text, "B2", scale_factor=1.0)
        
        high_entropy_label = Text("High Entropy", font_size=18, color=WHITE)
        self.place_in_area(high_entropy_label, "F1", "F3", scale_factor=1.0)

        # 20 uniform bars for SALET
        salet_bars = VGroup(*[
            Rectangle(
                width=0.08, 
                height=0.4 + np.random.uniform(0, 0.2), 
                fill_opacity=0.8, 
                fill_color="#83C167", 
                stroke_width=0.5
            ) 
            for _ in range(20)
        ]).arrange(RIGHT, buff=0.02)
        
        self.place_in_area(salet_bars, "C1", "E3", scale_factor=1.0)
        salet_bars.align_to(self.grid["E1"], DOWN)

        self.play(FadeIn(salet_text), Create(salet_bars), Write(high_entropy_label))

        # 50 dots for SALET (half of 100)
        salet_dots = VGroup(*[
            Dot(radius=0.03, color=WHITE) for _ in range(50)
        ])
        for dot in salet_dots:
            dot.move_to(self.grid["B2"] + np.array([np.random.uniform(-0.5, 0.5), 0, 0]))
        
        self.play(FadeIn(salet_dots))
        
        s_anims = []
        for i, dot in enumerate(salet_dots):
            bar_idx = i % 20
            target_pos = salet_bars[bar_idx].get_bottom() + np.array([
                0, 
                np.random.uniform(0.05, salet_bars[bar_idx].height - 0.05), 
                0
            ])
            s_anims.append(dot.animate.move_to(target_pos))
            
        self.play(*s_anims, run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # MUMMY label/asset and bars (Issue 33 & 46)
        mummy_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/mummy.svg")
        self.place_at_grid(mummy_asset, "B5", scale_factor=0.5)
        mummy_asset.set_color(WHITE)
        
        low_entropy_label = Text("Low Entropy", font_size=18, color=WHITE)
        self.place_in_area(low_entropy_label, "F4", "F6", scale_factor=1.0)

        # 1 tall bar, 19 tiny bars for MUMMY
        mummy_bars = VGroup(
            Rectangle(width=0.08, height=1.6, fill_opacity=0.8, fill_color="#58C4DD", stroke_width=0.5),
            *[Rectangle(width=0.08, height=0.08, fill_opacity=0.8, fill_color="#58C4DD", stroke_width=0.5) for _ in range(19)]
        ).arrange(RIGHT, buff=0.02)
        
        self.place_in_area(mummy_bars, "C4", "E6", scale_factor=1.0)
        mummy_bars.align_to(self.grid["E4"], DOWN)
        
        self.play(FadeIn(mummy_asset), Create(mummy_bars), Write(low_entropy_label))

        # 50 dots for MUMMY (other half of 100)
        mummy_dots = VGroup(*[
            Dot(radius=0.03, color=WHITE) for _ in range(50)
        ])
        for dot in mummy_dots:
            dot.move_to(self.grid["B5"] + np.array([np.random.uniform(-0.5, 0.5), 0, 0]))

        self.play(FadeIn(mummy_dots))
        
        m_anims = []
        for i, dot in enumerate(mummy_dots):
            if i < 40: # 80% to the big bucket
                target_pos = mummy_bars[0].get_bottom() + np.array([
                    0, 
                    np.random.uniform(0.1, 1.5), 
                    0
                ])
            else:
                idx = (i % 19) + 1
                target_pos = mummy_bars[idx].get_bottom() + np.array([0, 0.04, 0])
            m_anims.append(dot.animate.move_to(target_pos))
            
        self.play(*m_anims, run_time=1.5)
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
