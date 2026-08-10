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
        lecture_lines = [
            "Euler's formula relates graph parts.", 
            "Vertices minus edges plus faces equals two.", 
            "Works for any connected planar graph.", 
            "A triangle has three, three, two.", 
            "Resulting in two for all."
        ]
        self.setup_layout("Euler’s Characteristic Formula", lecture_lines)
        
        # Euler formula components
        formula = MathTex("V", "-", "E", "+", "F", "=", "2")
        formula[0].set_color("#FF5733") # V
        formula[2].set_color("#33FF57") # E
        formula[4].set_color("#3357FF") # F
        
        # Load triangle asset
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        
        formula_group = VGroup(formula, triangle).arrange(DOWN)
        self.place_in_area(formula_group, 'B4', 'C6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula), FadeIn(triangle))
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Highlighting logic handled by persistent colors as requested
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        values = MathTex("V=4, E=6, F=4")
        self.place_at_grid(values, 'B2', scale_factor=0.8)
        self.play(Write(values))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        self.play(Flash(formula[5:7]))
        
        # Move everything to a corner (F6 area)
        final_group = VGroup(formula_group, values)
        self.play(final_group.animate.scale(0.5).move_to(self.grid['F6']))
