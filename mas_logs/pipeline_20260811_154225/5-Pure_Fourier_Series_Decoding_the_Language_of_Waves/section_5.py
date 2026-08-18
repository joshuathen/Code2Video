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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We map signals from time into frequency domains.",
            "Noise-canceling headphones neutralize drone frequencies instantly.",
            "Fourier analysis transforms our digital, modern world."
        ]
        self.setup_layout("Conclusion & Application", lecture_lines)
        
        # Load Assets
        headphone_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/headphones.svg"
        drone_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg"
        
        headphone_icon = SVGMobject(headphone_path)
        drone_icon = SVGMobject(drone_path)
        
        # === Animation for Lecture Line 1 ===
        # Display real-world signal processing icons. (Color: #FFFFFF)
        icons = VGroup(headphone_icon, drone_icon).set_color("#FFFFFF")
        # Placing at A4 as per consolidation strategy
        self.place_at_grid(icons, 'A4', scale_factor=0.5)
        self.play(FadeIn(icons))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show signal decomposition. (Color: #00FFFF)
        decomp = VGroup(
            Line(ORIGIN, UP*1.5),
            Line(ORIGIN, RIGHT*1.5)
        ).set_color("#00FFFF")
        # Placing at C4 as per consolidation strategy
        self.place_at_grid(decomp, 'C4', scale_factor=0.6)
        self.play(Create(decomp))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Emphasize final spectrum output. (Color: #FFFF00)
        headphone_bar = SVGMobject(headphone_path).set_color("#FFFF00")
        bars = VGroup(
            Rectangle(height=1.0, width=0.2, color="#FFFF00", fill_opacity=1),
            Rectangle(height=1.5, width=0.2, color="#FFFF00", fill_opacity=1),
            Rectangle(height=0.8, width=0.2, color="#FFFF00", fill_opacity=1),
            headphone_bar
        ).arrange(DOWN, aligned_edge=DOWN, buff=0.1)
        
        # Placing at E4-F6 area
        self.place_in_area(bars, 'E4', 'F6', scale_factor=0.5)
        self.play(Create(bars))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
