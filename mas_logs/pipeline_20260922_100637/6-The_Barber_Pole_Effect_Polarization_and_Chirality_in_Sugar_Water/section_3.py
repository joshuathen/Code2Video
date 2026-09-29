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
        self.setup_layout("Mathematical Visualization: The Rotating Vector", [
            "Vector rotation represents light.", 
            "Angle θ depends on distance.", 
            "Different wavelengths rotate differently."
        ])
        
        # Load Assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        
        self.place_at_grid(laser, 'C1', scale_factor=0.3)
        self.place_at_grid(prism, 'D4', scale_factor=0.3)
        
        # Define objects
        vector = Arrow(ORIGIN, UP * 1.5, color="#00FFFF")
        formula = MathTex(r"\\theta = [\\alpha] \\cdot c \\cdot L", font_size=36)
        
        # Position objects based on feedback
        self.place_at_grid(vector, 'C2', scale_factor=0.7)
        self.place_at_grid(formula, 'B3', scale_factor=0.9)
        
        # Red and Blue wavelength indicators
        red_vec = Arrow(ORIGIN, UP * 1.2, color=RED)
        blue_vec = Arrow(ORIGIN, UP * 1.2, color=BLUE)
        self.place_at_grid(red_vec, 'D3', scale_factor=0.8)
        self.place_at_grid(blue_vec, 'D5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(laser), GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.play(Write(formula))
        self.play(Rotate(vector, angle=PI/2, about_point=vector.get_start()))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(FadeIn(prism), GrowArrow(red_vec), GrowArrow(blue_vec))
        self.play(
            Rotate(red_vec, angle=2*PI, rate_func=linear, run_time=3),
            Rotate(blue_vec, angle=4*PI, rate_func=linear, run_time=3)
        )
        self.wait(1)
