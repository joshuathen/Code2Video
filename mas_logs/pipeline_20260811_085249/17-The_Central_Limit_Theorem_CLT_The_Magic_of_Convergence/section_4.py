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
        lecture_lines = [
            "The CLT states sample means approach normality.",
            "This occurs regardless of the population distribution shape.",
            "Larger sample sizes yield a more perfect curve.",
            "The distribution always converges toward a normal shape.",
            "This bridge connects random data to normal statistics."
        ]
        self.setup_layout("The Central Limit Theorem Defined", lecture_lines)
        
        # Define objects
        pop_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        sample_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg")
        data_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/data.svg")
        
        pop_dist = VGroup(*[
            Rectangle(height=np.random.rand()*1.5, width=0.4, fill_opacity=0.8, color="#FFFFFF")
            for _ in range(10)
        ]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        clt_label = Text("CLT", font_size=72, weight=BOLD, color="#FF5733")
        
        # Stage objects - Using grid positions suggested by VideoCritic
        self.place_in_area(pop_dist, 'C2', 'E5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(pop_icon, 'B3', scale_factor=0.3)
        self.play(Create(pop_dist), FadeIn(pop_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_at_grid(data_icon, 'B5', scale_factor=0.3)
        self.play(Indicate(pop_dist), FadeIn(data_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#7FFF00")
        self.place_at_grid(clt_label, 'D3', scale_factor=0.7)
        self.play(FadeIn(clt_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF69B4")
        new_dist = VGroup(*[
            Rectangle(height=np.exp(-(i-5)**2/2), width=0.4, fill_opacity=0.8, color="#FF69B4")
            for i in range(10)
        ]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        self.place_in_area(new_dist, 'C2', 'E5', scale_factor=0.5)
        self.place_at_grid(sample_icon, 'D5', scale_factor=0.3)
        self.play(Transform(pop_dist, new_dist), FadeIn(sample_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        self.play(FadeOut(clt_label), FadeOut(pop_icon), FadeOut(data_icon), FadeOut(sample_icon))
        self.wait(2)
