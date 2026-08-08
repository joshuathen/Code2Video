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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-World Application", [
            "Holography is essentially frozen diffraction.",
            "It stores 3D spatial data efficiently.",
            "Modern applications include security and imaging."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Review holographic recording steps - represent as Interference pattern
        interference = VGroup(*[Circle(radius=0.1, color=BLUE).move_to(self.grid[f"{row}{col}"]) for row in "BCDE" for col in "2345"])
        # Fix 34 & 40: Adjust interference area
        self.place_in_area(interference, 'B4', 'E6', scale_factor=0.6)
        self.play(Create(interference))
        
        # Use Asset as requested in Issue 20
        hologram_icon_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        self.place_at_grid(hologram_icon_1, 'A5', scale_factor=0.5)
        self.play(FadeIn(hologram_icon_1))
        
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reconstruction process - two beams diverging
        beam1 = Line(start=self.grid['F1'], end=self.grid['A3'], color=GREEN)
        beam2 = Line(start=self.grid['F6'], end=self.grid['A3'], color=RED)
        reconstruction = VGroup(beam1, beam2)
        self.play(Create(reconstruction))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to Real-world - Credit card hologram
        hologram = Rectangle(width=3, height=2, color=YELLOW, fill_opacity=0.3)
        text_holo = Text("Hologram", font_size=24, color=YELLOW)
        card = VGroup(hologram, text_holo)
        
        # Use Asset as requested in Issue 20
        hologram_icon_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        card.add(hologram_icon_2)
        
        # Fix 33, 35 & 40: Adjust card position
        self.place_at_grid(card, 'F3', scale_factor=0.7)
        self.play(FadeIn(card))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
