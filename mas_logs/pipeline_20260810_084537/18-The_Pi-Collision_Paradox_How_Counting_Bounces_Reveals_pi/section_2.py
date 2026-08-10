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
        self.setup_layout("Prerequisites: Momentum and Energy", 
                          ["Collisions conserve both momentum and kinetic energy.", 
                           "Mapping velocities creates a 2D configuration space.", 
                           "This point moves steadily along the trajectory."])
        
        # Define mobjects
        momentum_formula = MathTex(r"p = mv", color="#00FFFF")
        energy_formula = MathTex(r"E = \frac{1}{2}mv^2", color="#FFFF00")
        billiards_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiards.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        # Address Issue 24: Use suggested grid positions and scale
        self.place_at_grid(momentum_formula, 'B2', scale_factor=0.7)
        self.place_at_grid(billiards_icon, 'B3', scale_factor=0.7)
        self.play(Write(momentum_formula), FadeIn(billiards_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Address Issue 24: Use suggested grid positions and scale
        self.place_at_grid(energy_formula, 'D2', scale_factor=0.7)
        self.play(Write(energy_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        combined_formulas = VGroup(momentum_formula, energy_formula, billiards_icon)
        # Address Issue 25: Use suggested area and scale
        self.place_in_area(combined_formulas, 'B4', 'E5', scale_factor=0.7)
        self.wait(2)
