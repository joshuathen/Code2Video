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
        self.setup_layout("Summary and Conclusion", [
            "Turbulence follows strict statistical laws.",
            "The 5/3 law represents turbulent DNA.",
            "Energy dissipation defines the cascade limit."
        ])
        
        # Elements
        stat_law = Text("Statistical Laws", font_size=36)
        dna_curve = MathTex(r"E(k) \propto k^{-5/3}", font_size=42)
        dissipation = Circle(radius=0.5, color=WHITE, fill_opacity=0.3)
        dissipation_label = Text("Dissipation", font_size=24)
        dissipation_group = VGroup(dissipation, dissipation_label).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(stat_law, 'B2')
        self.play(FadeIn(stat_law), self.lecture[0].animate.set_color("#FFD700"))
        self.play(stat_law.animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(dna_curve, 'C4')
        self.play(Write(dna_curve), self.lecture[1].animate.set_color("#FF4500"))
        self.play(dna_curve.animate.set_color("#FF4500"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(dissipation_group, 'E3')
        self.play(Create(dissipation), Write(dissipation_label), self.lecture[2].animate.set_color("#00BFFF"))
        self.play(dissipation.animate.set_color("#00BFFF"), dissipation_label.animate.set_color("#00BFFF"))
        self.wait(2)
