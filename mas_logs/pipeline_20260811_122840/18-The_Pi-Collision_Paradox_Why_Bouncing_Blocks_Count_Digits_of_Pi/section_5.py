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
        self.setup_layout("Summary & Synthesis", [
            "Discretized physics meets continuous geometry.", 
            "Complex dynamics reveal hidden mathematical constants.", 
            "Simple collisions map to deep numbers."
        ])
        
        # Load asset
        mass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mass.svg")
        
        # Define mobjects
        collision_visual = VGroup(
            mass_icon,
            Square(color=BLUE, fill_opacity=0.5).scale(0.5),
            Square(color=RED, fill_opacity=0.5).scale(0.8)
        ).arrange(RIGHT)
        
        pi_text = MathTex(r"\\pi \\approx 3.14159...").scale(1.2)
        
        # Groupings
        animation_group = VGroup(collision_visual, Arrow(), pi_text).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(animation_group, 'A4', 'F6', scale_factor=0.6)
        self.play(FadeIn(animation_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        pi_formula = pi_text.copy().set_color("#FF00FF")
        formula_box = SurroundingRectangle(pi_formula, color="#FF00FF", buff=0.2)
        
        self.play(
            FadeOut(collision_visual),
            FadeOut(animation_group[1]), # Arrow
            ReplacementTransform(pi_text, pi_formula)
        )
        self.place_at_grid(pi_formula, 'D4', scale_factor=0.7)
        self.place_in_area(formula_box, 'D3', 'E5', scale_factor=0.75)
        self.play(Create(formula_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        highlight = SurroundingRectangle(pi_formula, color="#FFFF00", buff=0.2)
        self.play(Create(highlight))
        self.wait(2)
