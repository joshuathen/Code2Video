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
        self.setup_layout("Mathematical Mechanics (The Sliding Dot Product)", [
            "Multiply filter values by underlying pixels.", 
            "Sum these values into one new pixel.", 
            "This defines a single discrete convolution."
        ])
        
        # Define objects
        formula = MathTex(r"\\sum", r"x_i", r"\\cdot", r"w_i").scale(1.2)
        
        # Assets
        pixel_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg")
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # B021: strict adherence. Used place_in_area as requested in Issue #24
        self.place_in_area(formula, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Grid and pixel setup
        self.place_at_grid(grid_asset, 'C3', scale_factor=0.5)
        self.place_at_grid(pixel_asset, 'C3', scale_factor=0.4)
        
        dots = VGroup(*[Dot(color=BLUE) for _ in range(3)])
        # B021: strict adherence. Used place_at_grid as requested in Issue #25
        self.place_at_grid(dots, 'E2', scale_factor=0.7)
        self.play(FadeIn(grid_asset), FadeIn(pixel_asset), FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        res = MathTex("0.5").set_color(GREEN)
        # B021: strict adherence. Used place_at_grid as requested in Issue #26
        self.place_at_grid(res, 'E5', scale_factor=1.0)
        self.play(Write(res))
        self.wait(2)
