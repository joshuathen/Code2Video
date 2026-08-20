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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Mechanism of Diffraction", [
            "Waves bend around small obstacles.",
            "Secondary wavelets form distinct patterns.",
            "Concentric rings reveal wave nature."
        ])
        
        # Elements
        slit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg", color="#CCCCCC")
        wavefronts = VGroup(*[Line(UP*0.5, DOWN*0.5, color="#FFFFFF") for _ in range(3)]).arrange(RIGHT, buff=0.3)
        wavelets = VGroup(*[Circle(radius=0.2, color="#FFFF00", fill_opacity=0.5) for _ in range(5)])
        screen = Rectangle(height=3, width=0.1, color="#FFFFFF", fill_opacity=1)
        
        diffraction_group = VGroup(slit, wavefronts, wavelets, screen)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#CCCCCC")
        # Position barrier
        self.place_at_grid(slit, 'C3', scale_factor=0.5)
        self.play(FadeIn(slit))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        # Illustrate wave
        self.place_at_grid(wavefronts, 'C2', scale_factor=0.8)
        self.play(Create(wavefronts))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Position and display rings
        self.place_at_grid(screen, 'C5', scale_factor=1.0)
        self.place_in_area(diffraction_group, 'B3', 'E5', scale_factor=0.9)
        self.play(FadeIn(wavelets), FadeIn(screen))
        self.wait(2)
