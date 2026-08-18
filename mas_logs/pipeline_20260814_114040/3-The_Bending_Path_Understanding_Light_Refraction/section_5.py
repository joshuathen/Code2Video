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
        self.setup_layout("Summary and Concept Check", ["Higher density slows light down.", "Light bends toward the Normal.", "Master refraction to understand light."])
        
        # Load Assets
        prism_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        lens_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        glass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        
        # Visual elements
        concept_table = VGroup(
            Text("Air to Glass (Slows)", font_size=20, color=YELLOW),
            Text("Glass to Air (Speeds up)", font_size=20, color=BLUE)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        arrow = Arrow(start=UP, end=DOWN, color=GREEN)
        formula = MathTex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fix for issue 35: place_at_grid(concept_table, 'A4', scale_factor=0.5)
        self.place_at_grid(concept_table, 'A4', scale_factor=0.5)
        self.place_at_grid(prism_icon, 'A6', scale_factor=0.4)
        self.play(FadeIn(concept_table), FadeIn(prism_icon))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        # Fix for issue 34: place_at_grid(arrow, 'B4', scale_factor=0.6)
        self.place_at_grid(arrow, 'B4', scale_factor=0.6)
        self.place_at_grid(lens_icon, 'B6', scale_factor=0.4)
        self.play(Create(arrow), FadeIn(lens_icon), Indicate(concept_table))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PURPLE)
        # Fix for issue 33: place_in_area(formula, 'B3', 'C6', scale_factor=0.9)
        self.place_in_area(formula, 'B3', 'C6', scale_factor=0.9)
        self.place_at_grid(glass_icon, 'D6', scale_factor=0.4)
        self.play(Write(formula), FadeIn(glass_icon))
        self.wait(2)
