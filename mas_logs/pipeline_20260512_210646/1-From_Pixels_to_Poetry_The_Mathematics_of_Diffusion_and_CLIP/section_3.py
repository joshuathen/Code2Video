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
        lecture_lines = [
            'To create, we must first understand how to destroy.',
            "Gaussian noise adds randomness to every pixel's value.",
            'This process increases entropy, erasing the original information.',
            'Eventually, the image becomes pure, undifferentiated static.'
        ]
        
        self.setup_layout("Prerequisite: Entropy and Gaussian Noise", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Owl Box (Issue 40/56: Area C2 to E5)
        owl_box = Rectangle(width=3.0, height=2.0, color=BLUE, stroke_width=2)
        self.place_in_area(owl_box, 'C2', 'E5')
        
        # Owl Icon Asset Integration (Issue 30/56: Asset owl.svg)
        owl_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/owl.svg")
        owl_icon.set_color(BLUE_A)
        self.place_in_area(owl_icon, 'C2', 'E5', scale_factor=0.6)
        
        # Owl Label (Issue 42/56: Area F2 to F3)
        owl_label = Text("Mechanical Owl", font_size=20, color=BLUE)
        self.place_in_area(owl_label, 'F2', 'F3', scale_factor=0.8)

        self.play(Create(owl_box), FadeIn(owl_icon), Write(owl_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Gaussian Bell Curve (Issue 41/56: Area A2 to B5)
        curve = FunctionGraph(
            lambda x: np.exp(-(x**2)) * 1.0,
            x_range=[-2, 2],
            color=WHITE
        )
        self.place_in_area(curve, 'A2', 'B5', scale_factor=0.7)

        # Noise grid setup overlayed on owl image area
        noise_grid = VGroup()
        rows_n, cols_n = 6, 8
        cell_w, cell_h = 3.0/cols_n, 2.0/rows_n
        for _ in range(rows_n * cols_n):
            sq = Rectangle(width=cell_w, height=cell_h, fill_opacity=0, stroke_width=0.1, stroke_color=GRAY_E)
            noise_grid.add(sq)
        noise_grid.arrange_in_grid(rows=rows_n, cols=cols_n, buff=0)
        self.place_in_area(noise_grid, 'C2', 'E5')

        self.play(Create(curve))
        
        # Adding randomness to pixel values
        indices = np.arange(len(noise_grid))
        np.random.shuffle(indices)
        batch1 = indices[:int(len(indices)*0.5)]
        
        noise_animations = []
        for idx in batch1:
            noise_animations.append(noise_grid[idx].animate.set_fill(color="#AAAAAA", opacity=np.random.uniform(0.4, 0.7)))
        
        self.play(*noise_animations, run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Transitioning information until it is random static
        self.play(FadeOut(curve))
        
        batch2 = indices[int(len(indices)*0.5):]
        noise_animations_2 = []
        for idx in batch2:
            noise_animations_2.append(noise_grid[idx].animate.set_fill(color=np.random.choice(["#AAAAAA", "#888888"]), opacity=0.8))
        for idx in batch1:
            noise_animations_2.append(noise_grid[idx].animate.set_fill(opacity=0.9))

        self.play(*noise_animations_2, run_time=1.5)
        self.play(FadeOut(owl_icon)) # Information content is erased
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        # Maximum Entropy Label (Issue 42/56: Area F4 to F5)
        entropy_label = Text("Maximum Entropy", font_size=20, color="#FF0000")
        self.place_in_area(entropy_label, 'F4', 'F5', scale_factor=0.8)

        # Static flicker effect
        flicker_batch = np.random.choice(len(noise_grid), 20, replace=False)
        flicker_anims = [noise_grid[idx].animate.set_fill(color=np.random.choice(["#CCCCCC", "#555555"])) for idx in flicker_batch]

        self.play(
            Write(entropy_label),
            *flicker_anims,
            run_time=1
        )
        self.wait(2)
