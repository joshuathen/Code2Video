from manim import *

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
        lecture_lines = ["A Bernoulli trial has exactly two outcomes.", "Success has probability p.", "Failure has probability 1-p."]
        self.setup_layout("Prerequisite Review: Bernoulli Trials", lecture_lines)
        
        # Create objects
        success_circle = Circle(radius=0.5, color="#00FF00", fill_opacity=0.5)
        success_text = Text("Success", font_size=20)
        failure_circle = Circle(radius=0.5, color="#FF0000", fill_opacity=0.5)
        failure_text = Text("Failure", font_size=20)
        
        p_label = Text("p", font_size=30, color="#00FF00")
        q_label = Text("1-p", font_size=30, color="#FF0000")
        
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        die_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg")
        
        # Position objects based on constraints
        self.place_at_grid(success_circle, 'B3', scale_factor=0.6)
        self.place_at_grid(success_text, 'C3', scale_factor=0.7)
        self.place_at_grid(failure_circle, 'B5', scale_factor=0.6)
        self.place_at_grid(failure_text, 'C5', scale_factor=0.7)
        
        # Place assets near the labels
        self.place_at_grid(coin_icon, 'A3', scale_factor=0.3)
        self.place_at_grid(die_icon, 'A5', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Create(success_circle), Write(success_text), Create(failure_circle), Write(failure_text))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        self.place_at_grid(p_label, 'A2', scale_factor=0.7)
        self.play(Write(p_label), FadeIn(coin_icon))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        self.place_at_grid(q_label, 'A4', scale_factor=0.7)
        self.play(Write(q_label), FadeIn(die_icon))
        
        self.wait(2)
