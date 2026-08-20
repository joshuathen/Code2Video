from manim import *
import numpy as np
import os

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
        self.setup_layout("Summary & Synthesis", [
            "Fourier series bridges time and frequency domains.",
            "Time domain shows waves changing over time.",
            "Frequency domain reveals the signal's core ingredients."
        ])
        
        # Paths
        radio_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/radio.svg"
        antenna_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg"

        # === Animation for Lecture Line 1 ===
        # Recap key concepts: Orthogonality and Decomposition using [Asset: radio.svg]. Color: #FFFFFF (white).
        self.lecture[0].set_color("#FFFFFF")
        
        # Asset Loading (Radio)
        radio_icon = SVGMobject(radio_path) if os.path.exists(radio_path) else Dot(color=WHITE)
        self.place_at_grid(radio_icon, 'B3', scale_factor=0.7)
        self.play(Create(radio_icon))

        # === Animation for Lecture Line 2 ===
        # Show signal transitioning back to components. Color: #00FFFF (cyan).
        self.lecture[1].set_color("#00FFFF")
        
        # Noisy wave animation
        wave = FunctionGraph(lambda x: np.sin(x) + 0.3*np.sin(3*x), x_range=[-2, 2])
        wave.set_stroke(color="#00FFFF")
        self.place_at_grid(wave, 'C5', scale_factor=0.6)
        self.play(Create(wave))

        # === Animation for Lecture Line 3 ===
        # Final display of Fourier series power via [Asset: antenna.svg]. Color: #FF0000 (red).
        self.lecture[2].set_color("#FF0000")
        
        # Asset Loading (Antenna)
        antenna_icon = SVGMobject(antenna_path) if os.path.exists(antenna_path) else Dot(color=RED)
        
        # Frequency bar chart
        bars = VGroup(*[Rectangle(height=1+i*0.5, width=0.3, fill_opacity=1, color=RED) for i in range(3)])
        bars.arrange(RIGHT, buff=0.1)
        self.place_at_grid(bars, 'D4', scale_factor=0.7)
        
        self.place_at_grid(antenna_icon, 'E4', scale_factor=0.5)
        self.play(Create(bars), Create(antenna_icon))
        self.wait(2)
