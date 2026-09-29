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
        self.setup_layout("Application: The Probabilistic Prediction Engine", 
                          ["LLMs predict the next token.", "Probabilities depend on prior context.", "'The' suggests 'cat' or 'car'."])
        
        # Prep assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Prep objects
        dist_bars = VGroup(*[Rectangle(height=1.5 * (0.8**i), width=0.8, color="#FFFFFF", fill_opacity=0.7) for i in range(5)])
        dist_bars.arrange(RIGHT, buff=0.2)
        
        picked_token = Text("cat", color="#FFFF00")
        next_word = Text("The cat", color="#800080")
        
        # Apply positioning constraints
        self.place_in_area(dist_bars, "B2", "B5", scale_factor=0.6)
        self.place_at_grid(picked_token, "D2", scale_factor=0.7)
        self.place_at_grid(next_word, "D5", scale_factor=0.7)
        self.place_at_grid(cat_icon, "C3", scale_factor=0.2)
        self.place_at_grid(car_icon, "C4", scale_factor=0.2)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(dist_bars), FadeIn(cat_icon), FadeIn(car_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(Indicate(dist_bars[0]), FadeIn(picked_token))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#800080")
        self.play(ReplacementTransform(picked_token, next_word))
        self.wait(1)
