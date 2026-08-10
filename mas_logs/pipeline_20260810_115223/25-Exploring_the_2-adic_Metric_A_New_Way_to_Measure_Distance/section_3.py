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
        self.setup_layout("Infinite Sums and Convergence", [
            "Convergence means terms approach zero in 2-adic.",
            "Higher powers of 2 stabilize the series.",
            "Adding powers of 2 creates binary towers."
        ])
        
        # Create elements
        series_text = MathTex("1 + 2 + 4 + 8 + \\dots", font_size=36)
        limit_text = MathTex("= -1", font_size=36, color="#00FFFF")
        
        # Assets
        tower_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.place_in_area(series_text, "B2", "B5", scale_factor=0.9)
        self.play(Write(series_text))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Visualization: tower for partial sums
        binary_tower = VGroup(tower_icon.copy(), Square(side_length=0.5, color=BLUE).shift(0.5*DOWN))
        self.place_in_area(binary_tower, "C2", "D4", scale_factor=0.5)
        self.play(Create(binary_tower))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # 2-adic limit marker
        limit_marker = VGroup(limit_text, tower_icon.copy().set_color("#00FFFF").scale(0.5))
        limit_marker.arrange(RIGHT)
        self.place_at_grid(limit_marker, "E3", scale_factor=0.8)
        self.play(Write(limit_marker))
        
        self.wait(2)
