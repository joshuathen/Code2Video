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
        lecture_lines = ["Formula: v_B equals P inverse v_A.", "P transforms new to old.", "Inverse converts old to new.", "A camera lens frame is new.", "Pixels map to world view."]
        self.setup_layout("The Change-of-Basis Formula", lecture_lines)
        
        # Elements
        formula = MathTex(r"[v]_B", "=", "P^{-1}", r"[v]_A", font_size=36)
        
        # Assets
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        pixels = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixels.svg")
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        arrow = Arrow(start=self.grid['C4'], end=self.grid['C5'], color=WHITE)
        eq_sign = formula[1]
        
        # Grid visual group (for criticism fix #24)
        grid_group = VGroup(formula, lens, pixels, camera, arrow)
        self.place_in_area(grid_group, 'A2', 'F6', scale_factor=0.9)
        
        # Formula placement (fix #25)
        self.place_in_area(formula, 'A4', 'C6', scale_factor=0.75)

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula))
        self.wait(1)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFA500")
        formula[2].set_color("#FFA500") # P
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#87CEFA")
        formula[2].set_color("#87CEFA") # Inverse P
        self.play(Indicate(formula[2])) 
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        self.place_at_grid(lens, 'D2', scale_factor=0.5)
        self.add(lens)
        self.play(FadeIn(lens))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#32CD32")
        self.place_at_grid(pixels, 'D4', scale_factor=0.4)
        self.place_at_grid(camera, 'D6', scale_factor=0.4)
        self.place_at_grid(arrow, 'D5', scale_factor=0.8) # Fix #23
        self.play(FadeIn(pixels), FadeIn(camera), GrowArrow(arrow))
        
        self.play(eq_sign.animate.set_color("#32CD32"))
        self.wait(2)
