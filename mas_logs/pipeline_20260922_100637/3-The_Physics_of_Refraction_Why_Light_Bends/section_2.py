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
        lecture_lines = ["Refractive index measures optical density.", 
                         "Think of light running through deep sand.", 
                         "Formula n equals c over v."]
        self.setup_layout("Defining Refractive Index (n)", lecture_lines)
        
        # Elements
        n1_label = Text("n₁", color="#FF00FF", font_size=36)
        n2_label = Text("n₂", color="#00FFFF", font_size=36)
        formula = MathTex(r"n = \frac{c}{v}", color=WHITE, font_size=48)
        sand_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        self.place_at_grid(n1_label, 'B2', scale_factor=0.9)
        self.place_at_grid(n2_label, 'B5', scale_factor=0.9)
        self.play(Write(n1_label), Write(n2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        # Visualizing 'running through sand' as a simple transition box
        rect = Rectangle(width=4, height=2, color=BLUE, fill_opacity=0.3)
        self.place_in_area(rect, 'C2', 'E5', scale_factor=0.85)
        self.place_at_grid(sand_icon, 'D4', scale_factor=0.5)
        self.play(FadeIn(rect), FadeIn(sand_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        self.place_at_grid(formula, 'D4', scale_factor=1.2)
        self.play(Write(formula))
        self.wait(2)
