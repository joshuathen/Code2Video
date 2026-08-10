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
        lecture_lines = ["Complex waves hide simple building blocks.", "A meow is just many sines.", "Watch these waves add up perfectly.", "Separate the waves into simple parts.", "Reconstructing the signal is our goal."]
        self.setup_layout("Intuitive Hook: The Symphony of Waves", lecture_lines)
        
        # Load assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        # Create elements
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3*x) + 0.3 * np.sin(5*x), color="#3498DB", x_range=[-3, 3])
        comp1 = FunctionGraph(lambda x: 0.5 * np.sin(3*x), color="#2ECC71", x_range=[-3, 3])
        comp2 = FunctionGraph(lambda x: 0.3 * np.sin(5*x), color="#2ECC71", x_range=[-3, 3])
        components = VGroup(comp1, comp2)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#3498DB")
        self.place_in_area(wave, 'B2', 'D4', scale_factor=0.5)
        self.place_at_grid(cat_icon, 'A5', scale_factor=0.3)
        self.play(Create(wave), FadeIn(cat_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E74C3C")
        self.play(wave.animate.set_color("#E74C3C"), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#ECF0F1")
        self.place_in_area(components, 'E2', 'F4', scale_factor=0.5)
        self.play(Create(components))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#2ECC71")
        self.play(wave.animate.set_stroke(opacity=0.5), run_time=1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F1C40F")
        self.play(FadeOut(components), wave.animate.set_stroke(opacity=1.0))
        self.wait(1)
