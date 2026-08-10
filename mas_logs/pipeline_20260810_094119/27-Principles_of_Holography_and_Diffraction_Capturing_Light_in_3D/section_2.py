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
        self.setup_layout("The Mechanism of Diffraction", ["Light spreads upon hitting obstacles.", "Huygens-Fresnel principle defines this scattering.", "Diffraction is an object's signature."])
        
        # === Animation for Lecture Line 1 ===
        # Place a point light source (white)
        light_source = Dot(color=WHITE)
        self.place_at_grid(light_source, 'D4', scale_factor=0.8)
        self.play(FadeIn(light_source))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Animate plane waves (cyan) hitting an obstacle
        diffraction_barrier = Line(start=self.grid['E5'], end=self.grid['F6'], color=GREY, stroke_width=6)
        self.place_in_area(diffraction_barrier, 'E5', 'F6', scale_factor=0.9)
        
        waves = VGroup(*[Circle(radius=0.5 + i*0.5, color="#00FFFF", stroke_width=2) for i in range(3)])
        waves.move_to(light_source.get_center())
        
        self.play(Create(diffraction_barrier))
        self.play(Create(waves))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Show waves bending around obstacle edges (orange)
        wave_arcs = VGroup(*[Arc(radius=0.5 + i*0.5, angle=PI/2, color=ORANGE, stroke_width=2) for i in range(3)])
        self.place_at_grid(wave_arcs, 'D5', scale_factor=0.7)
        
        self.play(ReplacementTransform(waves, wave_arcs))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
