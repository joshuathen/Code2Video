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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "2-adic distances follow the ultrametric inequality.",
            "Triangles behave more like nested hierarchies.",
            "Numerical value is secondary to divisibility."
        ]
        self.setup_layout("Key Properties: The Ultrametric Inequality", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color(YELLOW)
        
        formula = MathTex(r"|x + z|_2 \leq \max(|x|_2, |z|_2)")
        self.place_at_grid(formula, "C4", scale_factor=0.8)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Represent hierarchy with nested circles
        c1 = Circle(radius=2.0, color=BLUE).set_stroke(width=2)
        c2 = Circle(radius=1.2, color=GREEN).set_stroke(width=2)
        c3 = Circle(radius=0.6, color=RED).set_stroke(width=2)
        
        hierarchy = VGroup(c1, c2, c3)
        self.place_in_area(hierarchy, "D3", "F4", scale_factor=0.5)
        self.play(Create(c1), Create(c2), Create(c3))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Label to emphasize divisibility
        div_label = Text("Divisibility Rule", font_size=24, color=WHITE)
        self.place_at_grid(div_label, "F4", scale_factor=0.7)
        self.play(FadeIn(div_label))

        # Asset inclusion
        # SVG placeholder requested in storyboard, using generic shape as none.svg
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(asset_icon, "A2", scale_factor=0.5)
        self.play(FadeIn(asset_icon))

        self.wait(2)
