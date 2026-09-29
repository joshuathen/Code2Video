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
        self.setup_layout("Prerequisite: The Polarized Light Filter", [
            "Light behaves like a transverse wave.",
            "A polarizer acts as a picket fence.",
            "Only one polarization plane passes through."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Light behaves like a transverse wave.
        self.lecture[0].set_color("#FF9800")
        
        wave = VGroup(*[
            Line(UP * 0.5, DOWN * 0.5, color="#FF9800").shift(RIGHT * (i * 0.2))
            for i in range(-5, 6)
        ])
        self.place_in_area(wave, 'B3', 'D3', scale_factor=0.6)
        self.play(Create(wave))
        
        # === Animation for Lecture Line 2 ===
        # A polarizer acts as a picket fence.
        self.lecture[1].set_color("#FFEB3B")
        
        # Use SVG asset
        polarizer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fence.svg")
        polarizer.set_color("#FFEB3B")
        label_p = Text("Polarizer", color="#FFEB3B", font_size=20)
        polarizer_group = VGroup(polarizer, label_p).arrange(DOWN)
        self.place_in_area(polarizer_group, 'B4', 'D4', scale_factor=0.7)
        
        self.play(Create(polarizer_group))
        
        # === Animation for Lecture Line 3 ===
        # Only one polarization plane passes through.
        self.lecture[2].set_color("#2196F3")
        
        # Illustrate light passing through
        filtered_wave = Line(UP * 0.5, DOWN * 0.5, color="#2196F3")
        label_i = Text("Intensity", color="#2196F3", font_size=20)
        self.place_at_grid(filtered_wave, "C5")
        self.place_at_grid(label_i, 'D6', scale_factor=0.9)
        
        self.play(FadeIn(filtered_wave), Write(label_i))
        self.wait(2)
