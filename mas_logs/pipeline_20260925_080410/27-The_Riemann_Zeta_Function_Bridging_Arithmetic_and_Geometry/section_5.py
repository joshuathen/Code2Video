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
        self.setup_layout("Summary and Implications", ["Zeta bridges arithmetic and complex geometry.", "Solving it unlocks prime number secrets.", "A master key for mathematical structure."])
        
        # === Animation for Lecture Line 1 ===
        # Display 'Summary' in bold white (#FFFFFF) alongside [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg]
        summary_text = Text("Summary", font_size=36, color=WHITE)
        self.place_at_grid(summary_text, 'A2', scale_factor=1.0)
        
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg")
        self.place_at_grid(key_icon, 'A4', scale_factor=0.5)
        
        self.play(Write(summary_text), FadeIn(key_icon))
        self.play(self.lecture[0].animate.set_color("#D3D3D3"))
        
        # === Animation for Lecture Line 2 ===
        # List three core concepts in light gray (#D3D3D3)
        concept_1 = Text("Arithmetic-Geometry Bridge", font_size=20, color="#87CEEB")
        concept_2 = Text("Prime Mystery Key", font_size=20, color="#FFD700")
        concept_3 = Text("Mathematical Structure", font_size=20, color="#98FB98")
        
        self.place_in_area(concept_1, 'B3', 'B4', scale_factor=0.8)
        self.place_at_grid(concept_2, 'C3', scale_factor=0.8)
        self.place_in_area(concept_3, 'D3', 'D4', scale_factor=0.8)
        
        self.play(FadeIn(concept_1), FadeIn(concept_2), FadeIn(concept_3))
        self.play(self.lecture[1].animate.set_color("#D3D3D3"), self.lecture[2].animate.set_color("#D3D3D3"))
        
        # === Animation for Lecture Line 3 ===
        # Fade all elements out slowly
        self.wait(1)
        self.play(FadeOut(self.title), FadeOut(self.lecture), FadeOut(summary_text), FadeOut(key_icon), FadeOut(concept_1), FadeOut(concept_2), FadeOut(concept_3))
