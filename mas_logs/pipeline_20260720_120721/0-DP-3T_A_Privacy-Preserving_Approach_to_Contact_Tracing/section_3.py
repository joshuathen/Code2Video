from manim import *
import os
from pathlib import Path

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
            "When diagnosed, you report to the server.",
            "The server shares anonymized keys.",
            "These keys identify infected individuals."
        ]
        self.setup_layout("The Technology: Bluetooth & Cryptographic Keys", lecture_lines)
        
        # Define colors for lecture lines and corresponding animations
        color1 = "#FF6347" # Tomato
        color2 = "#4682B4" # SteelBlue
        color3 = "#3CB371" # MediumSeaGreen

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        # Text element
        report_text = Text("When diagnosed, you report to the server.", font_size=28, color=color1)
        self.place_at_grid(report_text, 'C4', scale_factor=0.7) # Place text at C4
        
        # Bluetooth icon asset
        bluetooth_icon_path = Path("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bluetooth.svg")
        bluetooth_icon = SVGMobject(str(bluetooth_icon_path)).set_color(color1)
        self.place_at_grid(bluetooth_icon, 'C3', scale_factor=0.6) # Place icon at C3, left of text

        self.play(FadeIn(report_text), FadeIn(bluetooth_icon))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        # Text element
        share_keys_text = Text("The server shares anonymized keys.", font_size=28, color=color2)
        self.place_at_grid(share_keys_text, 'D4', scale_factor=0.7) # Place at D4
        
        self.play(FadeIn(share_keys_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Text element
        identify_infected_text = Text("These keys identify infected individuals.", font_size=28, color=color3)
        self.place_at_grid(identify_infected_text, 'E4', scale_factor=0.7) # Place at E4
        
        self.play(FadeIn(identify_infected_text))
        self.wait(2)

        # Fade out all animation elements for a clean transition
        self.play(FadeOut(report_text, bluetooth_icon, share_keys_text, identify_infected_text))
        # Fade out the lecture lines and title at the very end
        self.play(FadeOut(self.lecture, self.title))
