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
        self.setup_layout("Visualizing Letter Frequency Distribution", [
            "English letters aren't equally common.", 
            "High-frequency letters narrow the search space.", 
            "'ARISE' eliminates more words than 'XYLYL'."
        ])
        
        # Frequency data (approximated for English)
        letters = ["E", "A", "R", "I", "O", "T", "N", "S", "L", "C", "U", "D", "P", "M", "H", "G", "B", "F", "Y", "W", "K", "V", "X", "Z", "J", "Q"]
        freqs = [12.7, 8.2, 7.5, 7.0, 7.5, 9.1, 6.7, 6.3, 4.0, 2.8, 2.8, 4.3, 1.9, 2.4, 6.1, 2.0, 1.5, 2.2, 2.0, 2.3, 0.8, 1.0, 0.15, 0.07, 0.15, 0.1]
        
        # B039: Unique colors
        bar_color = "#00FF00"
        highlight_color = "#FFFF00"
        
        # Create bar graph
        bars = VGroup()
        for i, val in enumerate(freqs):
            bar = Rectangle(height=val/15, width=0.15, fill_opacity=0.8, color=bar_color, fill_color=bar_color, stroke_width=0)
            label = Text(letters[i], font_size=10, color=WHITE)
            label.next_to(bar, DOWN, buff=0.05)
            bars.add(VGroup(bar, label))
        
        bars.arrange(RIGHT, buff=0.05)
        
        # Asset: Keyboard Icon
        keyboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg")
        keyboard_icon.set_color(WHITE)
        self.place_at_grid(keyboard_icon, "A6", scale_factor=0.3)
        
        # Chart title
        chart_title = Text("English Letter Frequencies", font_size=20, color=WHITE)
        self.place_in_area(chart_title, 'A2', 'A5', scale_factor=0.9)
        
        # Position bars
        self.place_in_area(bars, 'B3', 'E6', scale_factor=1.1)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(bar_color)
        self.play(Create(bars), FadeIn(keyboard_icon), FadeIn(chart_title), run_time=2)
        self.wait(1) 

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(bar_color)
        max_idx = np.argmax(freqs)
        self.play(bars[max_idx][0].animate.set_color(highlight_color), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(highlight_color)
        self.wait(2)
