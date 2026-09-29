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
        self.setup_layout("Summary & Takeaway", ["Fourier transforms deconstruct complex data.", "It translates time into spectrums.", "A powerful tool for analysis."])
        
        # Complex jumble of lines (Time)
        time_axes = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], axis_config={"include_tip": False}).scale(0.4)
        time_graph = time_axes.plot(lambda x: np.sin(x * 5) + 0.5 * np.sin(x * 15), color=BLUE)
        time_label = Text("Time Domain", font_size=20, color=BLUE).next_to(time_axes, UP)
        time_group = VGroup(time_axes, time_graph, time_label)
        # Fix 41/43: Applying area positioning and scaling
        self.place_in_area(time_group, 'A1', 'C3', scale_factor=0.6)
        
        # Frequency domain
        freq_axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], axis_config={"include_tip": False}).scale(0.4)
        # Using a simple bar for representation
        freq_bar1 = Rectangle(height=1.5, width=0.5, color=YELLOW, fill_opacity=1).next_to(freq_axes.c2p(1,0), UP, buff=0)
        freq_bar2 = Rectangle(height=0.8, width=0.5, color=YELLOW, fill_opacity=1).next_to(freq_axes.c2p(2.5,0), UP, buff=0)
        freq_label = Text("Frequency Domain", font_size=20, color=YELLOW).next_to(freq_axes, UP)
        freq_group = VGroup(freq_axes, freq_bar1, freq_bar2, freq_label)
        # Fix 42/43: Applying area positioning and scaling
        self.place_in_area(freq_group, 'A4', 'C6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(time_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(freq_group))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(Indicate(time_group), Indicate(freq_group))
        self.wait(2)
