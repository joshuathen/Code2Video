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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Archimedes discovered a remarkable geometric truth.",
            "The sphere's surface area is four pi r squared.",
            "This equals exactly four times the shadow's area.",
            "Visualize a cylinder wrapping around the sphere.",
            "The cylinder's surface area matches the sphere's."
        ]
        self.setup_layout("Linking Shadow to Surface Area", lecture_lines)
        
        # Assets (Using SVG if possible, falling back to basic shapes)
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=BLUE_D)
        cylinder = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cylinder.svg", color=YELLOW_D)
        shadow_circle = Circle(radius=1, color=WHITE, stroke_width=2)
        
        formula_surface = MathTex("A = 4\\pi r^2", color=ORANGE)
        formula_shadow = MathTex("A_{shadow} = \\pi r^2", color=TEAL)
        
        # Initial positions - Updated based on feedback
        self.place_at_grid(sphere, 'B3', scale_factor=0.8)
        self.place_at_grid(cylinder, 'B5', scale_factor=0.8)
        self.place_at_grid(shadow_circle, 'D3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(sphere))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(ORANGE))
        self.place_at_grid(formula_surface, 'C3', scale_factor=0.9)
        self.play(Write(formula_surface))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(TEAL))
        self.place_at_grid(formula_shadow, 'E3', scale_factor=0.9)
        self.play(Write(shadow_circle), Write(formula_shadow))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(Create(cylinder))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.play(Indicate(cylinder), sphere.animate.set_color("#FF00FF"))
