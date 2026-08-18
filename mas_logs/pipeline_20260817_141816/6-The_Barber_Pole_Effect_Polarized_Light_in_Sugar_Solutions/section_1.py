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
        lecture_lines = ["Polarized light vibrates in a single plane.", "A polarizer acts as a gate for light.", "Crossed polarizers block all light transmission completely."]
        self.setup_layout("Prerequisite: Polarization and Malus's Law", lecture_lines)
        
        # Load assets
        wave_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        wave = wave_asset.copy()
        wave.set_color(WHITE)
        self.place_in_area(wave, 'C1', 'E3', scale_factor=0.5)
        self.play(Create(wave))
        self.play(wave.animate.shift(RIGHT*2))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        polarizer = Rectangle(height=3, width=0.5, color=GREEN, fill_opacity=0.5)
        self.place_at_grid(polarizer, 'D2', scale_factor=0.7)
        self.play(Create(polarizer))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        polarizer2 = Rectangle(height=3, width=0.5, color=RED, fill_opacity=0.5)
        self.place_at_grid(polarizer2, 'D5', scale_factor=0.7)
        
        # Add Malus's Law curve asset as requested
        curve = wave_asset.copy()
        curve.set_color(YELLOW)
        self.place_at_grid(curve, 'F5', scale_factor=0.3)
        
        self.play(Create(polarizer2), Create(curve))
        self.play(FadeOut(wave))
