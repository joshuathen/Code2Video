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
            "Power, Root, and Log share numbers.",
            "They pivot around different unknown variables.",
            "Power finds y, Root finds b.",
            "Logarithm finds the exponent x."
        ]
        self.setup_layout("The Unified Notation Map", lecture_lines)
        
        # Elements
        triangle = Polygon(
            self.grid["B3"], self.grid["E1"], self.grid["E5"], 
            color=WHITE
        )
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] placeholder
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        label_p = MathTex("b^x = y", color=WHITE)
        label_r = MathTex("\\sqrt[x]{y} = b", color=WHITE)
        label_l = MathTex("\\log_b(y) = x", color=WHITE)
        
        formula_group = VGroup(triangle, label_p, label_r, label_l, asset_icon)
        self.place_in_area(formula_group, 'B2', 'E5', scale_factor=0.9)
        
        # Fixing anchoring per critic
        # Note: self.lecture is already managed by setup_layout which uses .to_edge(LEFT)
        # Applying requested fixes
        grid_visual_container = VGroup(formula_group)
        self.place_in_area(grid_visual_container, 'A3', 'F6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(triangle), FadeIn(label_p), self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(label_r), FadeIn(label_l), self.lecture[1].animate.set_color("#00CED1"))
        
        # === Animation for Lecture Line 3 ===
        arrow1 = Arrow(self.grid["B3"], self.grid["E1"], color="#FFD700")
        self.play(Create(arrow1), self.lecture[2].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 4 ===
        arrow2 = Arrow(self.grid["E1"], self.grid["E5"], color="#FFD700")
        self.play(Create(arrow2), self.lecture[3].animate.set_color("#FFD700"))
        
        self.wait(2)
