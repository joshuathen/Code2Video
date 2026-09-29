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
        self.setup_layout("Prerequisites & Intuition", [
            "Meet the normal distribution, defined by mean and variance.", 
            "Independent variables model separate events like morning routines.", 
            "Coffee time and commute time are independent random variables."
        ])
        
        # Create curves
        def get_normal(mu, sigma, color):
            return FunctionGraph(lambda x: np.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi)), x_range=[-3, 3], color=color)

        coffee_curve = get_normal(0, 0.5, BLUE)
        commute_curve = get_normal(0, 0.5, GREEN)
        
        coffee_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg", color=BLUE)
        house_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/house.svg", color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(coffee_curve, 'A3', 'B6', scale_factor=0.6)
        self.place_at_grid(coffee_icon, 'B3', scale_factor=0.5)
        self.play(Create(coffee_curve), FadeIn(coffee_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.place_in_area(commute_curve, 'D3', 'E6', scale_factor=0.6)
        self.play(Create(commute_curve))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PURPLE)
        self.place_at_grid(house_icon, 'E5', scale_factor=0.5)
        interaction = SurroundingRectangle(VGroup(coffee_curve, commute_curve), color=PURPLE, buff=0.2)
        self.play(Create(interaction), FadeIn(house_icon))
        self.wait(1)
