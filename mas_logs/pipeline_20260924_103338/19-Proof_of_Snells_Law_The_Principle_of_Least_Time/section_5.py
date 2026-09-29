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
        self.setup_layout("Conclusion & Real-world Application", [
            "Light follows nature's optimization principle.",
            "This rule explains fiber optics and mirages.",
            "Geometry reflects nature's efficient laws."
        ])
        
        # Asset Loading
        fiber_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fiber.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display Snell's Law label + Icon
        snells_formula = MathTex(r"n_1 \sin \theta_1 = n_2 \sin \theta_2", color=WHITE)
        self.place_in_area(snells_formula, 'B3', 'B6', scale_factor=0.9)
        self.place_at_grid(fiber_icon, 'D4', scale_factor=0.5)
        
        self.play(Write(snells_formula), FadeIn(fiber_icon))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # Animate refraction of light beam
        beam = VGroup(
            Line(start=self.grid['C2'], end=self.grid['D3'], color="#00FFFF"),
            Line(start=self.grid['D3'], end=self.grid['E5'], color="#00FFFF")
        )
        refraction_label = Text("Refraction", font_size=24, color="#00FFFF")
        self.place_at_grid(refraction_label, 'E4', scale_factor=0.7)
        
        self.play(Create(beam), Write(refraction_label))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        # Finalize with 'Snell's Law' text highlight
        self.play(snells_formula.animate.set_color(YELLOW))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
