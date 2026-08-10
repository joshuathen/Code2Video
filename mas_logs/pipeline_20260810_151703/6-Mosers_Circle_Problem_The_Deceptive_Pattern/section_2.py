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
            "One point gives one region.",
            "Two points give two regions.",
            "Three points give four regions.",
            "Four points give eight regions.",
            "Five points give sixteen regions."
        ]
        self.setup_layout("Data Collection & Pattern Hunting", lecture_lines)
        
        # Load asset
        circle_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg"
        circle = SVGMobject(circle_icon, color=WHITE)
        self.place_in_area(circle, 'A4', 'C6', scale_factor=0.6)
        self.add(circle)
        
        regions_text = MathTex(r"Regions: 1", color=WHITE)
        self.place_at_grid(regions_text, 'D4', scale_factor=0.9)
        self.add(regions_text)
        
        formula = MathTex(r"R = 2^{n-1}", color=WHITE)
        self.place_at_grid(formula, 'E4', scale_factor=1.0)
        self.add(formula)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"),
                  regions_text.animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"),
                  regions_text.animate.set_value(4))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"),
                  regions_text.animate.set_value(8))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"),
                  regions_text.animate.set_value(16),
                  formula.animate.set_color("#00FF00"))
        
        # Decoration for formula
        frame = SVGMobject(circle_icon, color="#FFFF00").scale(0.5).move_to(formula.get_center())
        self.play(Create(frame), formula.animate.set_color("#00FFFF"))
